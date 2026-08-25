#![allow(non_snake_case)]

use crate::algebra::{CscMatrix, ShapedMatrix, SparseFormatError};
use num_traits::Num;

/// Sparse matrix in Compressed Sparse Row (CSR) format
///
/// __Example usage__ : To construct the 3 x 3 matrix
/// ```text
/// A = [1.  3.  5.]
///     [2.  0.  6.]
///     [0.  4.  7.]
/// ```
/// ```no_run
/// use moreau::algebra::CsrMatrix;
///
/// let A : CsrMatrix<f64> = CsrMatrix::new(
///    3,                                // m
///    3,                                // n
///    vec![0, 3, 5, 7],                 // rowptr
///    vec![0, 1, 2, 0, 2, 1, 2],        // colval
///    vec![1., 3., 5., 2., 6., 4., 7.], // nzval
///  );
/// ```
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CsrMatrix<T = f64> {
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// CSR format row pointer.
    ///
    /// This field should have length `m+1`. The last entry corresponds
    /// to the number of nonzeros and should agree with the lengths
    /// of the `colval` and `nzval` fields.
    pub rowptr: Vec<usize>,
    /// vector of column indices
    pub colval: Vec<usize>,
    /// vector of non-zero matrix elements
    pub nzval: Vec<T>,
}

impl<T> CsrMatrix<T>
where
    T: Num + Copy,
{
    /// `CsrMatrix` constructor.
    ///
    /// # Panics
    /// Makes rudimentary dimensional compatibility checks and panics on failure.
    pub fn new(m: usize, n: usize, rowptr: Vec<usize>, colval: Vec<usize>, nzval: Vec<T>) -> Self {
        assert_eq!(colval.len(), nzval.len());
        assert_eq!(rowptr.len(), m + 1);
        assert_eq!(rowptr[m], colval.len());
        CsrMatrix {
            m,
            n,
            rowptr,
            colval,
            nzval,
        }
    }

    /// allocate space for a sparse matrix with `nnz` elements
    pub fn spalloc(size: (usize, usize), nnz: usize) -> Self {
        let (m, n) = size;
        let mut rowptr = vec![0; m + 1];
        let colval = vec![0; nnz];
        let nzval = vec![T::zero(); nnz];
        rowptr[m] = nnz;

        CsrMatrix::new(m, n, rowptr, colval, nzval)
    }

    /// Sparse matrix of zeros of size `m` x `n`
    pub fn zeros(size: (usize, usize)) -> Self {
        Self::spalloc(size, 0)
    }

    /// number of nonzeros
    pub fn nnz(&self) -> usize {
        self.rowptr[self.m]
    }

    /// Negate the matrix in place.
    pub fn negate(&mut self)
    where
        T: std::ops::Neg<Output = T>,
    {
        for v in &mut self.nzval {
            *v = -*v;
        }
    }

    /// Vertically concatenate two matrices.
    pub fn vcat(A: &Self, B: &Self) -> Result<Self, SparseFormatError>
    where
        T: Default,
    {
        if A.n != B.n {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        let m = A.m + B.m;
        let n = A.n;
        let nnz = A.nnz() + B.nnz();

        let mut rowptr = Vec::with_capacity(m + 1);
        let mut colval = Vec::with_capacity(nnz);
        let mut nzval = Vec::with_capacity(nnz);

        // Copy A's rowptr
        rowptr.extend_from_slice(&A.rowptr);
        colval.extend_from_slice(&A.colval);
        nzval.extend_from_slice(&A.nzval);

        // Append B's rowptr (offset by A's nnz)
        let offset = A.nnz();
        for &ptr in &B.rowptr[1..] {
            rowptr.push(ptr + offset);
        }
        colval.extend_from_slice(&B.colval);
        nzval.extend_from_slice(&B.nzval);

        Ok(CsrMatrix {
            m,
            n,
            rowptr,
            colval,
            nzval,
        })
    }

    /// Check that matrix data is dimensionally consistent.
    pub fn check_format(&self) -> Result<(), SparseFormatError> {
        if self.colval.len() != self.nzval.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        if self.rowptr.is_empty()
            || (self.rowptr.len() - 1) != self.m
            || self.rowptr[self.m] != self.colval.len()
        {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        // check for rowptr monotonicity
        if self.rowptr.windows(2).any(|c| c[0] > c[1]) {
            return Err(SparseFormatError::BadColptr); // Reuse error type
        }

        // check for column values out of bounds
        if !self.colval.iter().all(|c| c < &self.n) {
            return Err(SparseFormatError::BadRowval); // Reuse error type
        }

        Ok(())
    }

    /// Convert CSR matrix to CSC format.
    ///
    /// Returns a tuple of (CscMatrix, csr_to_csc_map) where:
    /// - The CscMatrix is the converted matrix
    /// - csr_to_csc_map[i] gives the index in CSC nzval for CSR nzval[i]
    ///
    /// This mapping is useful for converting derivative values back to CSR order.
    pub fn to_csc_with_mapping(&self) -> (CscMatrix<T>, Vec<usize>) {
        let nnz = self.nnz();

        // Count entries per column
        let mut colcounts = vec![0usize; self.n];
        for &col in &self.colval {
            colcounts[col] += 1;
        }

        // Build colptr from counts
        let mut colptr = vec![0usize; self.n + 1];
        for col in 0..self.n {
            colptr[col + 1] = colptr[col] + colcounts[col];
        }

        // Allocate rowval and nzval for CSC
        let mut rowval = vec![0usize; nnz];
        let mut nzval = vec![T::zero(); nnz];
        let mut csr_to_csc_map = vec![0usize; nnz];

        // Reset colcounts to use as write pointers
        colcounts.fill(0);

        // Fill CSC arrays
        for row in 0..self.m {
            for csr_idx in self.rowptr[row]..self.rowptr[row + 1] {
                let col = self.colval[csr_idx];
                let csc_idx = colptr[col] + colcounts[col];

                rowval[csc_idx] = row;
                nzval[csc_idx] = self.nzval[csr_idx];
                csr_to_csc_map[csr_idx] = csc_idx;

                colcounts[col] += 1;
            }
        }

        // Sort entries within each column by row index (CSC requires sorted rows)
        // This is needed because CSR order doesn't guarantee sorted column indices
        for col in 0..self.n {
            let start = colptr[col];
            let end = colptr[col + 1];
            if end > start + 1 {
                // Get indices that would sort by row
                let mut indices: Vec<usize> = (start..end).collect();
                indices.sort_by_key(|&i| rowval[i]);

                // Apply permutation
                let orig_rowval: Vec<_> = rowval[start..end].to_vec();
                let orig_nzval: Vec<_> = nzval[start..end].to_vec();

                for (new_pos, &old_pos) in indices.iter().enumerate() {
                    rowval[start + new_pos] = orig_rowval[old_pos - start];
                    nzval[start + new_pos] = orig_nzval[old_pos - start];
                }

                // Update the mapping to reflect the sorted positions
                // We need to update csr_to_csc_map for all CSR entries that map into this column
                for row in 0..self.m {
                    for csr_idx in self.rowptr[row]..self.rowptr[row + 1] {
                        if self.colval[csr_idx] == col {
                            // Find the new position of this row in the sorted CSC column
                            let target_row = row;
                            for (new_pos, &r) in rowval[start..end].iter().enumerate() {
                                if r == target_row {
                                    csr_to_csc_map[csr_idx] = start + new_pos;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        let csc = CscMatrix::new(self.m, self.n, colptr, rowval, nzval);

        (csc, csr_to_csc_map)
    }

    /// Convert CSR matrix to CSC format.
    pub fn to_csc(&self) -> CscMatrix<T> {
        self.to_csc_with_mapping().0
    }

    /// Create a CSR matrix from a CSC matrix.
    ///
    /// Returns a tuple of (CsrMatrix, csc_to_csr_map) where csc_to_csr_map[i]
    /// gives the index in CSR nzval for CSC nzval[i].
    pub fn from_csc_with_mapping(csc: &CscMatrix<T>) -> (Self, Vec<usize>) {
        let nnz = csc.nnz();

        // Count entries per row
        let mut rowcounts = vec![0usize; csc.m];
        for &row in &csc.rowval {
            rowcounts[row] += 1;
        }

        // Build rowptr from counts
        let mut rowptr = vec![0usize; csc.m + 1];
        for row in 0..csc.m {
            rowptr[row + 1] = rowptr[row] + rowcounts[row];
        }

        // Allocate colval and nzval for CSR
        let mut colval = vec![0usize; nnz];
        let mut nzval = vec![T::zero(); nnz];
        let mut csc_to_csr_map = vec![0usize; nnz];

        // Reset rowcounts to use as write pointers
        rowcounts.fill(0);

        // Fill CSR arrays
        for col in 0..csc.n {
            for csc_idx in csc.colptr[col]..csc.colptr[col + 1] {
                let row = csc.rowval[csc_idx];
                let csr_idx = rowptr[row] + rowcounts[row];

                colval[csr_idx] = col;
                nzval[csr_idx] = csc.nzval[csc_idx];
                csc_to_csr_map[csc_idx] = csr_idx;

                rowcounts[row] += 1;
            }
        }

        // Sort entries within each row by column index (CSR should have sorted columns)
        for row in 0..csc.m {
            let start = rowptr[row];
            let end = rowptr[row + 1];
            if end > start + 1 {
                // Get indices that would sort by column
                let mut indices: Vec<usize> = (start..end).collect();
                indices.sort_by_key(|&i| colval[i]);

                // Apply permutation
                let orig_colval: Vec<_> = colval[start..end].to_vec();
                let orig_nzval: Vec<_> = nzval[start..end].to_vec();

                for (new_pos, &old_pos) in indices.iter().enumerate() {
                    colval[start + new_pos] = orig_colval[old_pos - start];
                    nzval[start + new_pos] = orig_nzval[old_pos - start];
                }

                // Update the mapping
                for col in 0..csc.n {
                    for csc_idx in csc.colptr[col]..csc.colptr[col + 1] {
                        if csc.rowval[csc_idx] == row {
                            let target_col = col;
                            for (new_pos, &c) in colval[start..end].iter().enumerate() {
                                if c == target_col {
                                    csc_to_csr_map[csc_idx] = start + new_pos;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        let csr = CsrMatrix::new(csc.m, csc.n, rowptr, colval, nzval);

        (csr, csc_to_csr_map)
    }

    /// Create a CSR matrix from a CSC matrix.
    pub fn from_csc(csc: &CscMatrix<T>) -> Self {
        Self::from_csc_with_mapping(csc).0
    }

    /// Identity matrix of size `n`
    pub fn identity(n: usize) -> Self {
        let rowptr = (0usize..=n).collect();
        let colval = (0usize..n).collect();
        let nzval = vec![T::one(); n];

        CsrMatrix::new(n, n, rowptr, colval, nzval)
    }
}

/// Creates a CsrMatrix from a slice of arrays.
///
/// Example:
/// ```
/// use moreau::algebra::CsrMatrix;
/// let A = CsrMatrix::from(
///      &[[1.0, 2.0],
///        [3.0, 0.0],
///        [0.0, 4.0]]);
/// ```
impl<'a, I, J, T> From<I> for CsrMatrix<T>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = &'a T>,
    T: Num + Copy + 'a,
{
    #[allow(clippy::needless_range_loop)]
    fn from(rows: I) -> CsrMatrix<T> {
        let rows: Vec<Vec<T>> = rows
            .into_iter()
            .map(|r| r.into_iter().copied().collect())
            .collect();

        let m = rows.len();
        let n = rows.iter().map(|r| r.len()).next().unwrap_or(0);

        assert!(rows.iter().all(|r| r.len() == n));
        let nnz = rows.iter().flatten().filter(|&v| *v != T::zero()).count();

        let mut rowptr = Vec::with_capacity(m + 1);
        let mut colval = Vec::with_capacity(nnz);
        let mut nzval = Vec::<T>::with_capacity(nnz);

        rowptr.push(0);
        for r in 0..m {
            for c in 0..n {
                let value = rows[r][c];
                if value != T::zero() {
                    colval.push(c);
                    nzval.push(value);
                }
            }
            rowptr.push(nzval.len());
        }

        CsrMatrix::<T> {
            m,
            n,
            rowptr,
            colval,
            nzval,
        }
    }
}

impl<T> ShapedMatrix for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn size(&self) -> (usize, usize) {
        (self.m, self.n)
    }
    fn shape(&self) -> crate::algebra::MatrixShape {
        crate::algebra::MatrixShape::N
    }
    fn is_square(&self) -> bool {
        self.m == self.n
    }
}

/// Maps values from CSC order to CSR order using a precomputed mapping.
///
/// Given values in CSC order (matching a CscMatrix's nzval order) and
/// a csc_to_csr mapping, produces values in CSR order.
pub fn map_csc_to_csr_values<T: Copy>(csc_values: &[T], csc_to_csr_map: &[usize]) -> Vec<T> {
    if csc_values.is_empty() {
        return Vec::new();
    }
    let mut csr_values = vec![csc_values[0]; csc_values.len()]; // Use first value as placeholder
    for (csc_idx, &csr_idx) in csc_to_csr_map.iter().enumerate() {
        csr_values[csr_idx] = csc_values[csc_idx];
    }
    csr_values
}

/// Maps values from CSR order to CSC order using a precomputed mapping.
///
/// Given values in CSR order (matching a CsrMatrix's nzval order) and
/// a csr_to_csc mapping, produces values in CSC order.
pub fn map_csr_to_csc_values<T: Copy + Default>(
    csr_values: &[T],
    csr_to_csc_map: &[usize],
) -> Vec<T> {
    if csr_values.is_empty() {
        return Vec::new();
    }
    let mut csc_values = vec![csr_values[0]; csr_values.len()]; // Use first value as placeholder
    for (csr_idx, &csc_idx) in csr_to_csc_map.iter().enumerate() {
        csc_values[csc_idx] = csr_values[csr_idx];
    }
    csc_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_to_csc_roundtrip() {
        // Matrix:
        // [1  0  2]
        // [0  3  0]
        // [4  5  6]
        let csr = CsrMatrix::new(
            3,
            3,
            vec![0, 2, 3, 6],
            vec![0, 2, 1, 0, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        let (csc, _csr_to_csc) = csr.to_csc_with_mapping();

        // Verify CSC structure
        assert_eq!(csc.m, 3);
        assert_eq!(csc.n, 3);
        assert_eq!(csc.nnz(), 6);

        // Column 0: rows 0, 2 with values 1, 4
        // Column 1: rows 1, 2 with values 3, 5
        // Column 2: rows 0, 2 with values 2, 6
        assert_eq!(csc.colptr, vec![0, 2, 4, 6]);

        // Convert back to CSR
        let (csr2, csc_to_csr) = CsrMatrix::from_csc_with_mapping(&csc);

        assert_eq!(csr.m, csr2.m);
        assert_eq!(csr.n, csr2.n);
        assert_eq!(csr.rowptr, csr2.rowptr);
        assert_eq!(csr.colval, csr2.colval);
        assert_eq!(csr.nzval, csr2.nzval);

        // Test value mapping
        let csc_values = &csc.nzval;
        let mapped_back = map_csc_to_csr_values(csc_values, &csc_to_csr);
        assert_eq!(mapped_back, csr.nzval);
    }

    #[test]
    fn test_csr_construction() {
        let csr: CsrMatrix<f64> = CsrMatrix::new(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 2, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        );

        assert_eq!(csr.m, 2);
        assert_eq!(csr.n, 3);
        assert_eq!(csr.nnz(), 4);
        assert!(csr.check_format().is_ok());
    }

    #[test]
    fn test_empty_matrix() {
        let csr: CsrMatrix<f64> = CsrMatrix::zeros((3, 3));
        assert_eq!(csr.nnz(), 0);
        assert!(csr.check_format().is_ok());

        let csc = csr.to_csc();
        assert_eq!(csc.nnz(), 0);
    }

    #[test]
    fn test_gradient_mapping() {
        // Create a CSR matrix
        let csr = CsrMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        );

        // Convert to CSC and get mapping
        let (csc, _csr_to_csc) = csr.to_csc_with_mapping();

        // Simulate gradient computation that returns values in CSC order
        let csc_gradients = csc.nzval.iter().map(|v| v * 10.0).collect::<Vec<_>>();

        // Get the reverse mapping
        let (_, csc_to_csr) = CsrMatrix::from_csc_with_mapping(&csc);

        // Map gradients back to CSR order
        let csr_gradients = map_csc_to_csr_values(&csc_gradients, &csc_to_csr);

        // Verify the gradients correspond to the original CSR positions
        let expected: Vec<f64> = csr.nzval.iter().map(|v| v * 10.0).collect();
        assert_eq!(csr_gradients, expected);
    }
}
