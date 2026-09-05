#![allow(non_snake_case)]

use super::datamaps::*;
use crate::algebra::*;
use crate::solver::core::cones::*;
use num_traits::Zero;
use std::collections::BTreeSet;

pub(crate) fn allocate_kkt_Hsblocks<T, Z>(cones: &CompositeCone<T>) -> Vec<Z>
where
    T: FloatT,
    Z: Zero + Clone,
{
    let mut nnz = 0;
    if let Some(rng_last) = cones.rng_blocks.last() {
        nnz = rng_last.end;
    }
    vec![Z::zero(); nnz]
}

#[allow(unused_variables)]
pub(crate) fn assemble_kkt_matrix<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    shape: MatrixTriangle,
) -> (CscMatrix<T>, LDLDataMap) {
    // Slack-only assembly. Delegates to the full routine with an empty
    // direct-x composite cone — keeps the "no-xcones" path bit-identical
    // to before direct-x existed.
    let empty_xcones = CompositeXCone::<T>::new(&[]);
    assemble_kkt_matrix_full(P, A, cones, &empty_xcones, shape)
}

/// Assembly that handles both slack sparse expansions (SOC / GenPowerCone)
/// and direct-x sparse expansions (SOC, currently the only direct-x cone
/// type that uses rank-2 expansion). Direct-x sparse columns are placed
/// at the END of the KKT column range, AFTER all slack expansion
/// columns, in the same xcone order as `CompositeXCone::iter()`.
fn assemble_kkt_matrix_full<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    dir_cones: &CompositeXCone<T>,
    shape: MatrixTriangle,
) -> (CscMatrix<T>, LDLDataMap) {
    let mut map = LDLDataMap::new(P, A, cones);
    map.x_sparse_maps = (0..dir_cones.len()).map(|_| None).collect();

    let (m, n) = A.size();
    let p_slack = map.sparse_maps.pdim();
    let p_x = dir_cones.x_sparse_pdim();
    let p_total = p_slack + p_x;

    // LDLDataMap::new sized `diag_full` for (m + n + p_slack); grow it
    // now that we know p_x.
    map.diag_full.resize(m + n + p_total, 0);

    // entries actually on the diagonal of P.
    // NB: user provided P is always triu regardless
    // of the target shape of the KKT matrix
    let nnz_diagP = P.count_diagonal_entries(MatrixTriangle::Triu);

    // total entries in the Hs blocks
    let nnz_Hsblocks = map.Hsblocks.len();

    // Direct-x sparse expansion adds 2*(k+1) nnz per sparse-expandable
    // direct-x cone (k u-column entries + k v-column entries + 2 diag
    // in the 2 extra cols). Compute upfront so we can allocate the
    // right KKT size.
    let mut nnz_x_sparse = 0usize;
    for entry in dir_cones.iter() {
        if !entry.cone.direct_x_is_sparse_expandable() {
            continue;
        }
        match &entry.cone {
            SupportedCone::SecondOrderCone(_) => {
                // SOC rank-2: 2 cols × k entries + 2 diag = 2*(k+1).
                nnz_x_sparse += 2 * (entry.indices.len() + 1);
            }
            SupportedCone::GenPowerCone(c) => {
                // GenPow rank-9: q col (dim1) + r col (dim2) + p col (dim)
                // + 6 PD axis cols (each dim) + 9 diag.
                // = dim1 + dim2 + dim + 6·dim + 9 = 8·dim + 9.
                let dim = c.dim1() + c.dim2();
                nnz_x_sparse += 8 * dim + 9;
            }
            _ => {
                panic!("direct-x sparse expansion not implemented for this cone");
            }
        }
    }

    let nnzKKT = P.nnz() +      // Number of elements in P
    n -                         // Number of elements in diagonal top left block
    nnz_diagP +                 // remove double count on the diagonal if P has entries
    A.nnz() +                   // Number of nonzeros in A
    nnz_Hsblocks +              // Number of elements in diagonal below A'
    map.sparse_maps.nnz_vec() +  // Number of elements in sparse cone off diagonals
    p_slack +                   // diagonal of slack sparse cone expansion cols
    nnz_x_sparse;

    let dim = m + n + p_total;
    let mut K = CscMatrix::<T>::spalloc((dim, dim), nnzKKT);

    _kkt_assemble_colcounts(&mut K, P, A, cones, &map, shape);
    _x_sparse_colcounts(&mut K, dir_cones, m + n + p_slack, shape);
    _kkt_assemble_fill_with_xcones(
        &mut K,
        P,
        A,
        cones,
        dir_cones,
        m + n + p_slack,
        &mut map,
        shape,
    );

    (K, map)
}

/// Add colcount contributions for direct-x sparse expansion columns.
/// `pcol_start` is the first direct-x sparse column index (after all
/// slack sparse expansion columns).
fn _x_sparse_colcounts<T: FloatT>(
    K: &mut CscMatrix<T>,
    dir_cones: &CompositeXCone<T>,
    pcol_start: usize,
    shape: MatrixTriangle,
) {
    let mut pcol = pcol_start;
    for entry in dir_cones.iter() {
        if !entry.cone.direct_x_is_sparse_expandable() {
            continue;
        }
        match &entry.cone {
            SupportedCone::SecondOrderCone(_) => {
                // SOC rank-2: u col + v col, each `k` entries at
                // scattered x-indexed rows + 1 diag entry per col.
                let k = entry.indices.len();
                match shape {
                    MatrixTriangle::Triu => {
                        K.colptr[pcol] += k;
                        K.colptr[pcol + 1] += k;
                    }
                    MatrixTriangle::Tril => {
                        for &c in &entry.sorted_indices {
                            K.colptr[c] += 2;
                        }
                    }
                }
                K.colcount_diag(pcol, 2);
                pcol += 2;
            }
            SupportedCone::GenPowerCone(c) => {
                // GenPow rank-9: rank-3 base (q, r, p) + rank-6 PD axes
                // (each dim entries scattered into all cone indices). Plus
                // 9 diagonal entries. Mirrors slack
                // [`GenPowExpansionMap::csc_colcount_sparsecone`].
                let dim1 = c.dim1();
                let dim2 = c.dim2();
                let dim = dim1 + dim2;
                match shape {
                    MatrixTriangle::Triu => {
                        K.colptr[pcol] += dim1; // q col
                        K.colptr[pcol + 1] += dim2; // r col
                        K.colptr[pcol + 2] += dim; // p col
                        for k in 0..6 {
                            K.colptr[pcol + 3 + k] += dim; // PD axis k
                        }
                    }
                    MatrixTriangle::Tril => {
                        // In tril, expansion vectors appear as rows at
                        // pcol+i. Per cone-column count: q (1 if pos<dim1)
                        // + r (1 if pos>=dim1) + p (1 always) + 6 PD axes
                        // (1 each, all rows). Total per cone index: 1 + 1
                        // + 6 = 8 (q OR r contributes 1).
                        let cone_indices = &entry.indices;
                        for &c_idx in cone_indices {
                            K.colptr[c_idx] += 8;
                        }
                    }
                }
                K.colcount_diag(pcol, 9);
                pcol += 9;
            }
            _ => {
                panic!("direct-x sparse expansion not implemented for this cone");
            }
        }
    }
}

fn _kkt_assemble_colcounts<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &LDLDataMap,
    shape: MatrixTriangle,
) {
    let (m, n) = A.size();

    // use K.p to hold nnz entries in each
    // column of the KKT matrix
    K.colptr.fill(0);

    match shape {
        MatrixTriangle::Triu => {
            K.colcount_block(P, 0, MatrixShape::N);
            K.colcount_missing_diag(P, 0);
            K.colcount_block(A, n, MatrixShape::T);
        }
        MatrixTriangle::Tril => {
            K.colcount_missing_diag(P, 0);
            K.colcount_block(P, 0, MatrixShape::T);
            K.colcount_block(A, 0, MatrixShape::N);
        }
    }

    // track the next sparse column to fill (assuming triu fill)
    let mut pcol = m + n; //next sparse column to fill
    let mut sparse_map_iter = map.sparse_maps.iter();

    for (i, cone) in cones.iter().enumerate() {
        let row = cones.rng_cones[i].start + n;

        // add the Hs blocks in the lower right
        let blockdim = cone.numel();
        if cone.Hs_is_diagonal() {
            K.colcount_diag(row, blockdim);
        } else {
            K.colcount_dense_triangle(row, blockdim, shape);
        }

        //add sparse expansions columns for sparse cones
        if cone.is_sparse_expandable() {
            let sc = cone.to_sparse_expansion().unwrap();
            let thismap = sparse_map_iter.next().unwrap();
            sc.csc_colcount_sparsecone(thismap, K, row, pcol, shape);
            pcol += thismap.pdim();
        }
    }
}

fn _kkt_assemble_fill<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &mut LDLDataMap,
    shape: MatrixTriangle,
) {
    _kkt_assemble_fill_inner(K, P, A, cones, map, shape, None)
}

/// Variant of the fill pass that also fills direct-x sparse expansion
/// columns before the final `backshift_colptrs`. `x_sparse_info` pairs
/// (dir_cones, pcol_start) when present; None skips the direct-x pass.
fn _kkt_assemble_fill_with_xcones<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    dir_cones: &CompositeXCone<T>,
    pcol_x_start: usize,
    map: &mut LDLDataMap,
    shape: MatrixTriangle,
) {
    _kkt_assemble_fill_inner(K, P, A, cones, map, shape, Some((dir_cones, pcol_x_start)))
}

fn _kkt_assemble_fill_inner<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &mut LDLDataMap,
    shape: MatrixTriangle,
    x_sparse_info: Option<(&CompositeXCone<T>, usize)>,
) {
    let (m, n) = A.size();

    // cumsum total entries to convert to K.p
    K.colcount_to_colptr();

    match shape {
        MatrixTriangle::Triu => {
            K.fill_block(P, &mut map.P, 0, 0, MatrixShape::N);
            K.fill_missing_diag(P, 0); // after adding P, since triu form
                                       // fill in value for A, top right (transposed/rowwise)
            K.fill_block(A, &mut map.A, 0, n, MatrixShape::T);
        }
        MatrixTriangle::Tril => {
            K.fill_missing_diag(P, 0); // before adding P, since tril form
            K.fill_block(P, &mut map.P, 0, 0, MatrixShape::T);
            // fill in value for A, bottom left (not transposed)
            K.fill_block(A, &mut map.A, n, 0, MatrixShape::N);
        }
    }

    // track the next sparse column to fill (assuming triu fill)
    let mut pcol = m + n; //next sparse column to fill
    let mut sparse_map_iter = map.sparse_maps.iter_mut();

    for (i, cone) in cones.iter().enumerate() {
        let row = cones.rng_cones[i].start + n;

        // add the Hs blocks in the lower right
        let blockdim = cone.numel();
        let block = &mut map.Hsblocks[cones.rng_blocks[i].clone()];

        if cone.Hs_is_diagonal() {
            K.fill_diag(block, row, blockdim);
        } else {
            K.fill_dense_triangle(block, row, blockdim, shape);
        }

        //add sparse expansions columns for sparse cones
        if cone.is_sparse_expandable() {
            let sc = cone.to_sparse_expansion().unwrap();
            let thismap = sparse_map_iter.next().unwrap();
            sc.csc_fill_sparsecone(thismap, K, row, pcol, shape);
            pcol += thismap.pdim();
        }
    }

    // Direct-x sparse expansion — must run BEFORE `backshift_colptrs`
    // so we can continue using the advancing colptr as the fill cursor.
    if let Some((dir_cones, pcol_x_start)) = x_sparse_info {
        let mut pcol_x = pcol_x_start;
        for (xi, entry) in dir_cones.iter().enumerate() {
            if !entry.cone.direct_x_is_sparse_expandable() {
                continue;
            }
            match &entry.cone {
                SupportedCone::SecondOrderCone(_) => {
                    let k = entry.indices.len();
                    let mut x_map = DirectXSparseMapSOC::new(k, entry.cone_pos_for_sorted.clone());
                    // Slack SOC rank-2 layout: FIRST extra col carries
                    // `v`-scaled values (diag `-η²` slack, `+η²` direct-x);
                    // SECOND carries `u`-scaled values (diag flipped).
                    match shape {
                        MatrixTriangle::Triu => {
                            K.fill_colvec_scattered(&mut x_map.v, &entry.sorted_indices, pcol_x);
                            K.fill_colvec_scattered(
                                &mut x_map.u,
                                &entry.sorted_indices,
                                pcol_x + 1,
                            );
                        }
                        MatrixTriangle::Tril => {
                            K.fill_rowvec_scattered(&mut x_map.v, pcol_x, &entry.indices);
                            K.fill_rowvec_scattered(&mut x_map.u, pcol_x + 1, &entry.indices);
                        }
                    }
                    K.fill_diag(&mut x_map.D, pcol_x, 2);
                    map.x_sparse_maps[xi] = Some(DirectXSparseMap::SOC(x_map));
                    pcol_x += 2;
                }
                SupportedCone::GenPowerCone(c) => {
                    let dim1 = c.dim1();
                    let dim2 = c.dim2();
                    // Build sub-permutations: split entry.indices into
                    // p-block (first dim1) and w-block (last dim2),
                    // sort each, and record original cone-internal position.
                    let cone_indices = &entry.indices;
                    let mut dim1_pairs: Vec<(usize, usize)> =
                        (0..dim1).map(|i| (cone_indices[i], i)).collect();
                    dim1_pairs.sort_by_key(|&(v, _)| v);
                    let sorted_dim1: Vec<usize> = dim1_pairs.iter().map(|&(v, _)| v).collect();
                    let cone_pos_for_sorted_dim1: Vec<usize> =
                        dim1_pairs.iter().map(|&(_, i)| i).collect();

                    let mut dim2_pairs: Vec<(usize, usize)> = (0..dim2)
                        .map(|i| (cone_indices[dim1 + i], dim1 + i))
                        .collect();
                    dim2_pairs.sort_by_key(|&(v, _)| v);
                    let sorted_dim2: Vec<usize> = dim2_pairs.iter().map(|&(v, _)| v).collect();
                    let cone_pos_for_sorted_dim2: Vec<usize> =
                        dim2_pairs.iter().map(|&(_, i)| i).collect();

                    let mut x_map = DirectXSparseMapGenPow::new(
                        dim1,
                        dim2,
                        cone_pos_for_sorted_dim1,
                        cone_pos_for_sorted_dim2,
                        entry.cone_pos_for_sorted.clone(),
                    );
                    // GenPow rank-9 layout: q col (dim1, scattered into
                    // p-block indices), r col (dim2, scattered into w-block
                    // indices), p col (dim, scattered into all indices),
                    // then 6 PD axes (each dim, scattered into all indices)
                    // mirroring slack's [`GenPowExpansionMap`].
                    let dim1_cone_indices: Vec<usize> = cone_indices[..dim1].to_vec();
                    let dim2_cone_indices: Vec<usize> = cone_indices[dim1..].to_vec();
                    match shape {
                        MatrixTriangle::Triu => {
                            K.fill_colvec_scattered(&mut x_map.q, &sorted_dim1, pcol_x);
                            K.fill_colvec_scattered(&mut x_map.r, &sorted_dim2, pcol_x + 1);
                            K.fill_colvec_scattered(
                                &mut x_map.p_v,
                                &entry.sorted_indices,
                                pcol_x + 2,
                            );
                            for k in 0..6 {
                                K.fill_colvec_scattered(
                                    &mut x_map.pd_axes[k],
                                    &entry.sorted_indices,
                                    pcol_x + 3 + k,
                                );
                            }
                        }
                        MatrixTriangle::Tril => {
                            K.fill_rowvec_scattered(&mut x_map.q, pcol_x, &dim1_cone_indices);
                            K.fill_rowvec_scattered(&mut x_map.r, pcol_x + 1, &dim2_cone_indices);
                            K.fill_rowvec_scattered(&mut x_map.p_v, pcol_x + 2, cone_indices);
                            for k in 0..6 {
                                K.fill_rowvec_scattered(
                                    &mut x_map.pd_axes[k],
                                    pcol_x + 3 + k,
                                    cone_indices,
                                );
                            }
                        }
                    }
                    K.fill_diag(&mut x_map.D, pcol_x, 9);
                    map.x_sparse_maps[xi] = Some(DirectXSparseMap::GenPow(x_map));
                    pcol_x += 9;
                }
                _ => {
                    panic!("direct-x sparse expansion not implemented for this cone");
                }
            }
        }
    }

    // backshift the colptrs to recover K.p again
    K.backshift_colptrs();

    // Now we can populate the index of the full diagonal.
    // We have filled in structural zeros on it everywhere.

    match shape {
        MatrixTriangle::Triu => {
            // matrix is triu, so diagonal is last in each column
            map.diag_full.copy_from_slice(&K.colptr[1..]);
            map.diag_full.iter_mut().for_each(|x| *x -= 1);
            // and the diagonal of just the upper left
            map.diagP.copy_from_slice(&K.colptr[1..=n]);
            map.diagP.iter_mut().for_each(|x| *x -= 1);
        }

        MatrixTriangle::Tril => {
            // matrix is tril, so diagonal is first in each column
            map.diag_full
                .copy_from_slice(&K.colptr[0..K.colptr.len() - 1]);
            // and the diagonal of just the upper left
            map.diagP.copy_from_slice(&K.colptr[0..n]);
        }
    }
}

// ----------------------------------------------------------------------
// Direct-x cone assembly
//
// The (1,1) KKT block becomes `P + Σ_J E_J' H_J E_J`, where `J` ranges
// over direct-x cones and `E_J` is the scatter from a k-vector into
// `x[indices]`. Structurally this means the (1,1) block pattern is
// `structure(P) ∪ ⋃_J {(i,j) : i,j ∈ indices_J, i <= j}`.
//
// We precompute an "extended P" whose structure is this union and whose
// values are P's values (zeros at new positions). Running the standard
// assembly with Pext gives us:
//   - A translation map from user-P entries to KKT positions.
//   - A translation map from each direct-x Hs block entry to its KKT
//     position — possibly sharing positions with user-P entries when the
//     direct-x footprint overlaps P.
// ----------------------------------------------------------------------

/// Returns `(Pext, p_to_pext, hx_to_pext)` where:
///
/// - `Pext` is a CscMatrix with structure = `structure(P) ∪ direct-x
///   footprint (triu)`. Original P values are preserved; new positions
///   hold structural zeros.
/// - `p_to_pext[k]` gives the position in `Pext.nzval` of the original
///   `P.nzval[k]`.
/// - `hx_to_pext[i]` gives the position in `Pext.nzval` of the `i`-th
///   entry of the stacked Hx buffer. The ordering matches iterating each
///   direct-x cone's `get_Hs` layout: diagonal cones produce `ni` entries
///   `(indices[p], indices[p])`; non-diagonal cones produce
///   `triangular_number(ni)` entries in column-major upper triangular
///   order over `indices`.
///
/// Caller supplies a user P matrix in triu form. `Pext` is also triu.
pub(crate) fn build_P_ext<T: FloatT>(
    P: &CscMatrix<T>,
    dir_cones: &CompositeXCone<T>,
) -> (CscMatrix<T>, Vec<usize>, Vec<usize>) {
    let n = P.nrows();
    assert_eq!(P.ncols(), n, "P must be square for build_P_ext");

    // Footprint positions in triu form, stored as (col, row) with row <= col,
    // in the cone-then-block order expected by the stacked Hx buffer.
    let mut hx_triu_positions: Vec<(usize, usize)> = Vec::with_capacity(dir_cones.hx_block_len());
    for entry in dir_cones.iter() {
        let indices = &entry.indices;
        let ni = indices.len();
        if entry.cone.direct_x_Hs_is_diagonal() {
            for p in 0..ni {
                let v = indices[p];
                hx_triu_positions.push((v, v));
            }
        } else {
            // column-major upper triangular: (r_logical, c_logical) with r_logical <= c_logical
            for c_logical in 0..ni {
                for r_logical in 0..=c_logical {
                    let a = indices[r_logical];
                    let b = indices[c_logical];
                    let (row, col) = if a <= b { (a, b) } else { (b, a) };
                    hx_triu_positions.push((col, row));
                }
            }
        }
    }

    // Per-column sorted set of rows (union of P rows and direct-x rows).
    let mut col_rows: Vec<BTreeSet<usize>> = (0..n).map(|_| BTreeSet::new()).collect();
    for j in 0..n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            col_rows[j].insert(P.rowval[k]);
        }
    }
    for &(col, row) in &hx_triu_positions {
        col_rows[col].insert(row);
    }

    // Build Pext's CSC arrays.
    let mut colptr = vec![0usize; n + 1];
    for j in 0..n {
        colptr[j + 1] = colptr[j] + col_rows[j].len();
    }
    let nnz_ext = colptr[n];
    let mut rowval = vec![0usize; nnz_ext];
    for j in 0..n {
        let mut off = colptr[j];
        for &r in &col_rows[j] {
            rowval[off] = r;
            off += 1;
        }
    }
    let mut nzval = vec![T::zero(); nnz_ext];

    // Copy P values and build p_to_pext.
    let mut p_to_pext = vec![0usize; P.nnz()];
    for j in 0..n {
        let col_start = colptr[j];
        let col_end = colptr[j + 1];
        for k in P.colptr[j]..P.colptr[j + 1] {
            let r = P.rowval[k];
            let rel = rowval[col_start..col_end]
                .binary_search(&r)
                .expect("P row must be present in Pext column");
            let pext_idx = col_start + rel;
            p_to_pext[k] = pext_idx;
            nzval[pext_idx] = P.nzval[k];
        }
    }

    // Build hx_to_pext.
    let mut hx_to_pext = vec![0usize; hx_triu_positions.len()];
    for (i, &(col, row)) in hx_triu_positions.iter().enumerate() {
        let col_start = colptr[col];
        let col_end = colptr[col + 1];
        let rel = rowval[col_start..col_end]
            .binary_search(&row)
            .expect("direct-x row must be present in Pext column");
        hx_to_pext[i] = col_start + rel;
    }

    let Pext = CscMatrix::new(n, n, colptr, rowval, nzval);
    (Pext, p_to_pext, hx_to_pext)
}

/// Variant of [`assemble_kkt_matrix`] that additionally supports direct-x
/// cone contributions to the (1,1) block. When `dir_cones` is empty this is
/// identical to `assemble_kkt_matrix`.
pub(crate) fn assemble_kkt_matrix_with_xcones<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    dir_cones: &CompositeXCone<T>,
    shape: MatrixTriangle,
) -> (CscMatrix<T>, LDLDataMap) {
    if dir_cones.is_empty() {
        return assemble_kkt_matrix(P, A, cones, shape);
    }

    let (Pext, p_to_pext, hx_to_pext) = build_P_ext(P, dir_cones);
    let (K, pext_map) = assemble_kkt_matrix_full(&Pext, A, cones, dir_cones, shape);

    // Translate Pext-indexed P map into user-P-indexed map, and build
    // Hxblocks by redirecting hx_to_pext through pext_map.P.
    let user_P: Vec<usize> = p_to_pext.iter().map(|&i| pext_map.P[i]).collect();
    let user_Hxblocks: Vec<usize> = hx_to_pext.iter().map(|&i| pext_map.P[i]).collect();

    // Invert p_to_pext so we can look up, per Hx entry, whether it coincides
    // with a user-P entry. `p_to_pext` is injective (each user-P nzval maps
    // to a unique Pext slot), so a flat Vec<Option<usize>> indexed by Pext
    // position is enough.
    let mut pext_to_p: Vec<Option<usize>> = vec![None; Pext.nnz()];
    for (k, &pext_idx) in p_to_pext.iter().enumerate() {
        pext_to_p[pext_idx] = Some(k);
    }
    let user_Hx_to_P: Vec<Option<usize>> = hx_to_pext.iter().map(|&i| pext_to_p[i]).collect();

    let LDLDataMap {
        P: _,
        A,
        Hsblocks,
        sparse_maps,
        Hxblocks: _,
        Hx_to_P: _,
        x_sparse_maps,
        diagP,
        diag_full,
    } = pext_map;

    let user_map = LDLDataMap {
        P: user_P,
        A,
        Hsblocks,
        sparse_maps,
        Hxblocks: user_Hxblocks,
        Hx_to_P: user_Hx_to_P,
        x_sparse_maps,
        diagP,
        diag_full,
    };

    (K, user_map)
}

#[test]
fn test_kkt_assembly_upper_lower() {
    let P = CscMatrix::from(&[
        [1., 2., 4.], //
        [0., 3., 5.], //
        [0., 0., 6.], //
    ]);
    let A = CscMatrix::from(&[
        [7., 0., 8.],  //
        [0., 9., 10.], //
        [1., 2., 3.],
        [7., 0., 8.],  //
        [0., 9., 10.], //
        [1., 2., 3.],
    ]);

    let Ku_true_nncone = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1., 7., 0., 1.],   //
        [0., 3., 5., 0., 9., 2., 0., 9., 2.],   //
        [0., 0., 6., 8., 10., 3., 8., 10., 3.], //
        [0., 0., 0., -1., 0., 0., 0., 0., 0.],  //
        [0., 0., 0., 0., -1., 0., 0., 0., 0.],  //
        [0., 0., 0., 0., 0., -1., 0., 0., 0.],  //
        [0., 0., 0., 0., 0., 0., -1., 0., 0.],  //
        [0., 0., 0., 0., 0., 0., 0., -1., 0.],  //
        [0., 0., 0., 0., 0., 0., 0., 0., -1.],  //
    ]);

    let Kl_true_nncone = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],   //
        [2., 3., 0., 0., 0., 0., 0., 0., 0.],   //
        [4., 5., 6., 0., 0., 0., 0., 0., 0.],   //
        [7., 0., 8., -1., 0., 0., 0., 0., 0.],  //
        [0., 9., 10., 0., -1., 0., 0., 0., 0.], //
        [1., 2., 3., 0., 0., -1., 0., 0., 0.],  //
        [7., 0., 8., 0., 0., 0., -1., 0., 0.],  //
        [0., 9., 10., 0., 0., 0., 0., -1., 0.], //
        [1., 2., 3., 0., 0., 0., 0., 0., -1.],  //
    ]);

    let Ku_true_expcones = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1., 7., 0., 1.],    //
        [0., 3., 5., 0., 9., 2., 0., 9., 2.],    //
        [0., 0., 6., 8., 10., 3., 8., 10., 3.],  //
        [0., 0., 0., -1., -1., -1., 0., 0., 0.], //
        [0., 0., 0., 0., -1., -1., 0., 0., 0.],  //
        [0., 0., 0., 0., 0., -1., 0., 0., 0.],   //
        [0., 0., 0., 0., 0., 0., -1., -1., -1.], //
        [0., 0., 0., 0., 0., 0., 0., -1., -1.],  //
        [0., 0., 0., 0., 0., 0., 0., 0., -1.],   //
    ]);

    let Kl_true_expcones = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],    //
        [2., 3., 0., 0., 0., 0., 0., 0., 0.],    //
        [4., 5., 6., 0., 0., 0., 0., 0., 0.],    //
        [7., 0., 8., -1., 0., 0., 0., 0., 0.],   //
        [0., 9., 10., -1., -1., 0., 0., 0., 0.], //
        [1., 2., 3., -1., -1., -1., 0., 0., 0.], //
        [7., 0., 8., 0., 0., 0., -1., 0., 0.],   //
        [0., 9., 10., 0., 0., 0., -1., -1., 0.], //
        [1., 2., 3., 0., 0., 0., -1., -1., -1.], //
    ]);

    // diagonal lower right block tests
    // --------------------------------
    let K = SupportedConeT::NonnegativeConeT(6);
    let cones = CompositeCone::new(&[K]);

    let (mut Ku, mapu) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Triu);
    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_nncone);

    let (mut Kl, mapl) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Tril);
    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_nncone);

    // dense blocks lower right block tests
    // --------------------------------
    let K = SupportedConeT::ExponentialConeT();
    let cones = CompositeCone::new(&[K.clone(), K.clone()]);

    let (mut Ku, mapu) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Triu);
    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_expcones);

    let (mut Kl, mapl) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Tril);
    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_expcones);
}

#[cfg(test)]
mod xcone_assembly_tests {
    use super::*;

    fn empty_xcones() -> CompositeXCone<f64> {
        CompositeXCone::<f64>::new(&[])
    }

    #[test]
    fn with_xcones_empty_matches_standard() {
        // When dir_cones is empty, the _with_xcones variant must produce
        // an identical KKT matrix and map (P/Hsblocks/A indices unchanged).
        let P = CscMatrix::from(&[
            [1., 2., 0.], //
            [0., 3., 4.], //
            [0., 0., 5.], //
        ]);
        let A = CscMatrix::from(&[
            [7., 0., 8.],  //
            [0., 9., 10.], //
        ]);
        let cones = CompositeCone::new(&[SupportedConeT::NonnegativeConeT(2)]);

        let (K_std, map_std) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Triu);
        let (K_xc, map_xc) =
            assemble_kkt_matrix_with_xcones(&P, &A, &cones, &empty_xcones(), MatrixTriangle::Triu);

        assert_eq!(K_std, K_xc);
        assert_eq!(map_std.P, map_xc.P);
        assert_eq!(map_std.A, map_xc.A);
        assert_eq!(map_std.Hsblocks, map_xc.Hsblocks);
        assert_eq!(map_std.diagP, map_xc.diagP);
        assert_eq!(map_std.diag_full, map_xc.diag_full);
        assert!(map_xc.Hxblocks.is_empty());
    }

    #[test]
    fn build_P_ext_no_overlap_nonneg() {
        // P is empty on some x-indices; direct-x nonneg adds new diagonal
        // entries at those positions.
        //     P = diag(1, 0, 5);  xcone = nonneg on [1]
        // Expected Pext diagonal: entries at (0,0), (1,1) (new), (2,2).
        let P = CscMatrix::from(&[
            [1., 0., 0.], //
            [0., 0., 0.], //
            [0., 0., 5.], //
        ]);
        let dir_cones = CompositeXCone::<f64>::new(&[SupportedXConeT::NonnegativeXConeT(vec![1])]);
        let (Pext, p_to_pext, hx_to_pext) = build_P_ext(&P, &dir_cones);

        // P.nnz() was 2, Pext should have 3.
        assert_eq!(Pext.nnz(), 3);
        assert_eq!(p_to_pext.len(), 2);
        assert_eq!(hx_to_pext.len(), 1);

        // Pext[(0,0)] = 1, Pext[(1,1)] = 0 (structural zero), Pext[(2,2)] = 5.
        let expected = CscMatrix::from(&[
            [1., 0., 0.], //
            [0., 0., 0.], //
            [0., 0., 5.], //
        ]);
        // Structurally Pext differs from P (has (1,1) now), even though value is 0.
        assert_eq!(Pext.nzval, vec![1.0, 0.0, 5.0]);
        assert_eq!(Pext.rowval, vec![0, 1, 2]);
        // Pext as a dense matrix equals P (values), since the new entry is 0.
        let mut dense_ext = vec![0.0; 9];
        for j in 0..3 {
            for k in Pext.colptr[j]..Pext.colptr[j + 1] {
                dense_ext[Pext.rowval[k] * 3 + j] = Pext.nzval[k];
            }
        }
        let mut dense_orig = vec![0.0; 9];
        for j in 0..3 {
            for k in expected.colptr[j]..expected.colptr[j + 1] {
                dense_orig[expected.rowval[k] * 3 + j] = expected.nzval[k];
            }
        }
        assert_eq!(dense_ext, dense_orig);
    }

    #[test]
    fn build_P_ext_overlap_nonneg() {
        // Nonneg direct-x on index 0; P already has an entry at (0,0).
        // hx_to_pext must collide with p_to_pext[that entry].
        let P = CscMatrix::from(&[
            [1., 0.], //
            [0., 2.], //
        ]);
        let dir_cones = CompositeXCone::<f64>::new(&[SupportedXConeT::NonnegativeXConeT(vec![0])]);
        let (Pext, p_to_pext, hx_to_pext) = build_P_ext(&P, &dir_cones);

        assert_eq!(Pext.nnz(), 2);
        // P's (0,0) entry is at index 0 in P; Pext preserves it.
        assert_eq!(Pext.nzval, vec![1.0, 2.0]);
        // Nonneg direct-x Hs entry for index 0 maps to Pext's (0,0) position.
        assert_eq!(hx_to_pext[0], p_to_pext[0]);
    }

    #[test]
    fn assemble_kkt_with_nonneg_xcone_diagonal_layout() {
        // Build a KKT with a direct-x nonneg cone and check that writing
        // identity Hs values via map.Hxblocks produces the expected dense
        // matrix: P + diag(1) on the direct-x indices.
        let P = CscMatrix::from(&[
            [1., 0., 0.], //
            [0., 0., 0.], //
            [0., 0., 5.], //
        ]);
        // Single equality constraint so that Hsblocks are non-empty.
        let A = CscMatrix::from(&[[1., 1., 1.]]);
        let cones = CompositeCone::new(&[SupportedConeT::<f64>::ZeroConeT(1)]);
        // Direct-x nonneg on x[1] and x[2]
        let dir_cones = CompositeXCone::<f64>::new(&[SupportedXConeT::NonnegativeXConeT(vec![1, 2])]);

        let (mut K, map) =
            assemble_kkt_matrix_with_xcones(&P, &A, &cones, &dir_cones, MatrixTriangle::Triu);

        // Hxblocks has 2 entries (diagonal nonneg, size 2).
        assert_eq!(map.Hxblocks.len(), 2);

        // Write Hs=1 at both direct-x positions. x[2] overlaps P[(2,2)]=5
        // in the user-visible sense only if we were additively accumulating;
        // our contract is that Hxblocks writes replace the nzval at that
        // KKT slot. For commit 3 we only check that Hxblocks points to the
        // expected KKT positions (the numeric baseline is commit 4's job).
        for i in &map.Hxblocks {
            K.nzval[*i] = 1.0;
        }

        // Dense reconstruction of upper triangle of K.
        let dim = 4; // n=3 + m=1
        let mut dense = vec![0.0; dim * dim];
        for j in 0..dim {
            for k in K.colptr[j]..K.colptr[j + 1] {
                let r = K.rowval[k];
                dense[r * dim + j] = K.nzval[k];
                if r != j {
                    dense[j * dim + r] = K.nzval[k];
                }
            }
        }

        // After Hxblocks write, K[(1,1)] = 1 (was structural zero from Pext),
        // K[(2,2)] = 1 (overwrote P's 5 at that slot). K[(0,0)] = 1 from P.
        assert_eq!(dense[1 * dim + 1], 1.0);
        assert_eq!(dense[2 * dim + 2], 1.0);
        assert_eq!(dense[0 * dim + 0], 1.0);
        // A' block is at top-right.
        assert_eq!(dense[0 * dim + 3], 1.0);
        assert_eq!(dense[1 * dim + 3], 1.0);
        assert_eq!(dense[2 * dim + 3], 1.0);
    }

    #[test]
    fn assemble_kkt_with_nonneg_xcone_tril() {
        // Same layout check for Tril KKT.
        let P = CscMatrix::from(&[
            [1., 0.], //
            [0., 0.], //
        ]);
        let A = CscMatrix::from(&[[1., 1.]]);
        let cones = CompositeCone::new(&[SupportedConeT::<f64>::ZeroConeT(1)]);
        let dir_cones = CompositeXCone::<f64>::new(&[SupportedXConeT::NonnegativeXConeT(vec![1])]);

        let (mut K, map) =
            assemble_kkt_matrix_with_xcones(&P, &A, &cones, &dir_cones, MatrixTriangle::Tril);

        assert_eq!(map.Hxblocks.len(), 1);
        for i in &map.Hxblocks {
            K.nzval[*i] = 7.0;
        }

        // Confirm (1,1) in the tril KKT got the value.
        let dim = 3; // n=2 + m=1
        let mut dense = vec![0.0; dim * dim];
        for j in 0..dim {
            for k in K.colptr[j]..K.colptr[j + 1] {
                let r = K.rowval[k];
                dense[r * dim + j] = K.nzval[k];
                if r != j {
                    dense[j * dim + r] = K.nzval[k];
                }
            }
        }
        assert_eq!(dense[1 * dim + 1], 7.0);
    }

    #[test]
    fn hx_to_p_overlap_and_gaps() {
        // P has entries at (0,0) and (2,2); direct-x nonneg touches
        // indices [0, 1, 2]. Only positions 0 and 2 overlap P; position
        // 1 is a new diagonal slot with no P counterpart.
        let P = CscMatrix::from(&[
            [1., 0., 0.], //
            [0., 0., 0.], //
            [0., 0., 5.], //
        ]);
        let A = CscMatrix::from(&[[1., 1., 1.]]);
        let cones = CompositeCone::new(&[SupportedConeT::<f64>::ZeroConeT(1)]);
        let dir_cones =
            CompositeXCone::<f64>::new(&[SupportedXConeT::NonnegativeXConeT(vec![0, 1, 2])]);

        let (_K, map) =
            assemble_kkt_matrix_with_xcones(&P, &A, &cones, &dir_cones, MatrixTriangle::Triu);

        assert_eq!(map.Hx_to_P.len(), 3);
        // Hx entry 0 → P's (0,0), which is the first user-P nzval
        assert_eq!(map.Hx_to_P[0], Some(0));
        // Hx entry 1 → new slot, no P overlap
        assert_eq!(map.Hx_to_P[1], None);
        // Hx entry 2 → P's (2,2), which is the second user-P nzval
        assert_eq!(map.Hx_to_P[2], Some(1));
    }

    #[test]
    fn assemble_kkt_with_two_nonneg_xcones() {
        // Two disjoint direct-x cones; check stacked Hxblocks ordering.
        let P = CscMatrix::from(&[
            [0., 0., 0., 0.], //
            [0., 0., 0., 0.], //
            [0., 0., 0., 0.], //
            [0., 0., 0., 0.], //
        ]);
        let A = CscMatrix::from(&[[1., 1., 1., 1.]]);
        let cones = CompositeCone::new(&[SupportedConeT::<f64>::ZeroConeT(1)]);
        let dir_cones = CompositeXCone::<f64>::new(&[
            SupportedXConeT::NonnegativeXConeT(vec![0, 1]),
            SupportedXConeT::NonnegativeXConeT(vec![2, 3]),
        ]);

        let (mut K, map) =
            assemble_kkt_matrix_with_xcones(&P, &A, &cones, &dir_cones, MatrixTriangle::Triu);
        assert_eq!(map.Hxblocks.len(), 4);

        // Write distinguishable values to each Hxblocks slot.
        K.nzval[map.Hxblocks[0]] = 10.0;
        K.nzval[map.Hxblocks[1]] = 20.0;
        K.nzval[map.Hxblocks[2]] = 30.0;
        K.nzval[map.Hxblocks[3]] = 40.0;

        let dim = 5;
        let mut dense = vec![0.0; dim * dim];
        for j in 0..dim {
            for k in K.colptr[j]..K.colptr[j + 1] {
                let r = K.rowval[k];
                dense[r * dim + j] = K.nzval[k];
            }
        }
        // Cone 1 writes to (0,0) and (1,1) in Hxblocks[0..2].
        // Cone 2 writes to (2,2) and (3,3) in Hxblocks[2..4].
        assert_eq!(dense[0 * dim + 0], 10.0);
        assert_eq!(dense[1 * dim + 1], 20.0);
        assert_eq!(dense[2 * dim + 2], 30.0);
        assert_eq!(dense[3 * dim + 3], 40.0);
    }
}
