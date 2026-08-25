#![allow(non_snake_case)]

use crate::algebra::CsrMatrix;
use core::ops::Deref;
use pyo3::prelude::*;

/// Python wrapper for CsrMatrix.
///
/// We can't implement the foreign trait FromPyObject directly on CsrMatrix
/// since it is outside the crate, so put a dummy wrapper around it here.
#[pyclass]
pub struct PyCsrMatrix(CsrMatrix<f64>);

impl Deref for PyCsrMatrix {
    type Target = CsrMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PyCsrMatrix> for CsrMatrix<f64> {
    fn from(mat: PyCsrMatrix) -> Self {
        mat.0
    }
}

impl From<CsrMatrix<f64>> for PyCsrMatrix {
    fn from(mat: CsrMatrix<f64>) -> Self {
        PyCsrMatrix(mat)
    }
}

impl<'a> FromPyObject<'a> for PyCsrMatrix {
    fn extract_bound(obj: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        // Validate that the matrix is in CSR format
        let format: String = obj.getattr("format")?.extract()?;
        if format != "csr" {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Expected CSR matrix (scipy.sparse.csr_matrix), got '{}' format. \
                 Convert with: matrix.tocsr()",
                format
            )));
        }

        // Extract CSR matrix components from scipy.sparse.csr_matrix
        // scipy uses: data, indices (column indices), indptr (row pointers)
        let nzval: Vec<f64> = obj.getattr("data")?.extract()?;
        let colval: Vec<usize> = obj.getattr("indices")?.extract()?;
        let rowptr: Vec<usize> = obj.getattr("indptr")?.extract()?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        let mat = CsrMatrix::new(shape[0], shape[1], rowptr, colval, nzval);

        Ok(PyCsrMatrix(mat))
    }
}
