#![allow(non_snake_case)]
#![allow(clippy::new_without_default)]

use crate::solver::core::cones::{SupportedConeT, SupportedConeT::*, SupportedXConeT};
use pyo3::{exceptions::PyTypeError, prelude::*};
use std::fmt::Write;

// generic Python display functionality for cone objects
fn __repr__cone(name: &str, dim: usize) -> String {
    let mut s = String::new();
    write!(s, "{}({})", name, dim).unwrap();
    s
}

// generic Python display functionality for cone objects
// with no parameters, specifically 3d expcone
fn __repr__cone__noparams(name: &str) -> String {
    let mut s = String::new();
    write!(s, "{}()", name).unwrap();
    s
}

// generic Python display functionality for cone objects
// with floating point parameters (specifically power cone)
fn __repr__cone__float(name: &str, pow: f64) -> String {
    let mut s = String::new();
    write!(s, "{}({})", name, pow).unwrap();
    s
}

// Python display functionality for genpowercone objects
// with floating point vector parameters
fn __repr__genpowcone(name: &str, alpha: &[f64], dim2: usize) -> String {
    let mut s = String::new();
    write!(s, "{}[\n    α = {:?},\n dim2 = {}\n]", name, alpha, dim2).unwrap();
    s
}

#[pyclass(name = "ZeroConeT")]
pub struct PyZeroConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PyZeroConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("ZeroConeT", self.dim)
    }
}

#[pyclass(name = "NonnegativeConeT")]
pub struct PyNonnegativeConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PyNonnegativeConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("NonnegativeConeT", self.dim)
    }
}

#[pyclass(name = "SecondOrderConeT")]
pub struct PySecondOrderConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PySecondOrderConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("SecondOrderConeT", self.dim)
    }
}

#[pyclass(name = "ExponentialConeT")]
pub struct PyExponentialConeT {}
#[pymethods]
impl PyExponentialConeT {
    #[new]
    pub fn new() -> Self {
        Self {}
    }
    pub fn __repr__(&self) -> String {
        __repr__cone__noparams("ExponentialConeT")
    }
}

#[pyclass(name = "PowerConeT")]
pub struct PyPowerConeT {
    #[pyo3(get)]
    pub α: f64,
}
#[pymethods]
impl PyPowerConeT {
    #[new]
    pub fn new(α: f64) -> Self {
        Self { α }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone__float("PowerConeT", self.α)
    }
}

#[pyclass(name = "GenPowerConeT")]
pub struct PyGenPowerConeT {
    #[pyo3(get)]
    pub α: Vec<f64>,
    #[pyo3(get)]
    pub dim2: usize,
}
#[pymethods]
impl PyGenPowerConeT {
    #[new]
    pub fn new(α: Vec<f64>, dim2: usize) -> Self {
        Self { α, dim2 }
    }
    pub fn __repr__(&self) -> String {
        __repr__genpowcone("GenPowerConeT", &self.α, self.dim2)
    }
}

#[pyclass(name = "PSDTriangleConeT")]
pub struct PyPSDTriangleConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PyPSDTriangleConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("PyPSDTriangleConeT", self.dim)
    }
}

// -------------------------------------
// Direct-x cone wrappers.
// -------------------------------------

#[pyclass(name = "NonnegativeXConeT")]
pub struct PyNonnegativeXConeT {
    #[pyo3(get)]
    pub indices: Vec<usize>,
}
#[pymethods]
impl PyNonnegativeXConeT {
    #[new]
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }
    pub fn __repr__(&self) -> String {
        format!("NonnegativeXConeT({:?})", self.indices)
    }
}

#[pyclass(name = "SecondOrderXConeT")]
pub struct PySecondOrderXConeT {
    #[pyo3(get)]
    pub indices: Vec<usize>,
}
#[pymethods]
impl PySecondOrderXConeT {
    #[new]
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }
    pub fn __repr__(&self) -> String {
        format!("SecondOrderXConeT({:?})", self.indices)
    }
}

#[pyclass(name = "ExponentialXConeT")]
pub struct PyExponentialXConeT {
    #[pyo3(get)]
    pub indices: Vec<usize>,
}
#[pymethods]
impl PyExponentialXConeT {
    #[new]
    pub fn new(indices: Vec<usize>) -> PyResult<Self> {
        if indices.len() != 3 {
            return Err(PyTypeError::new_err(format!(
                "ExponentialXConeT requires exactly 3 indices, got {}",
                indices.len()
            )));
        }
        Ok(Self { indices })
    }
    pub fn __repr__(&self) -> String {
        format!("ExponentialXConeT({:?})", self.indices)
    }
}

#[pyclass(name = "PowerXConeT")]
pub struct PyPowerXConeT {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub alpha: f64,
}
#[pymethods]
impl PyPowerXConeT {
    #[new]
    pub fn new(indices: Vec<usize>, alpha: f64) -> PyResult<Self> {
        if indices.len() != 3 {
            return Err(PyTypeError::new_err(format!(
                "PowerXConeT requires exactly 3 indices, got {}",
                indices.len()
            )));
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(PyTypeError::new_err(format!(
                "PowerXConeT alpha must be in (0, 1), got {}",
                alpha
            )));
        }
        Ok(Self { indices, alpha })
    }
    pub fn __repr__(&self) -> String {
        format!("PowerXConeT({:?}, alpha={})", self.indices, self.alpha)
    }
}

#[pyclass(name = "GenPowerXConeT")]
pub struct PyGenPowerXConeT {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub alphas: Vec<f64>,
    #[pyo3(get)]
    pub dim2: usize,
}
#[pymethods]
impl PyGenPowerXConeT {
    #[new]
    pub fn new(indices: Vec<usize>, alphas: Vec<f64>, dim2: usize) -> PyResult<Self> {
        if alphas.is_empty() {
            return Err(PyTypeError::new_err(
                "GenPowerXConeT requires alphas of length >= 1".to_string(),
            ));
        }
        if dim2 < 1 {
            return Err(PyTypeError::new_err(format!(
                "GenPowerXConeT requires dim2 >= 1, got {}",
                dim2
            )));
        }
        if indices.len() != alphas.len() + dim2 {
            return Err(PyTypeError::new_err(format!(
                "GenPowerXConeT indices.len() = {} but alphas.len() + dim2 = {} + {} = {}",
                indices.len(),
                alphas.len(),
                dim2,
                alphas.len() + dim2
            )));
        }
        for (i, &a) in alphas.iter().enumerate() {
            if !(a > 0.0) {
                return Err(PyTypeError::new_err(format!(
                    "GenPowerXConeT alphas[{}] must be > 0, got {}",
                    i, a
                )));
            }
        }
        let asum: f64 = alphas.iter().sum();
        if (asum - 1.0).abs() > 1e-8 * (alphas.len() as f64) {
            return Err(PyTypeError::new_err(format!(
                "GenPowerXConeT alphas must sum to 1, got {}",
                asum
            )));
        }
        Ok(Self {
            indices,
            alphas,
            dim2,
        })
    }
    pub fn __repr__(&self) -> String {
        format!(
            "GenPowerXConeT({:?}, alphas={:?}, dim2={})",
            self.indices, self.alphas, self.dim2
        )
    }
}

#[cfg(feature = "sdp")]
#[pyclass(name = "PSDTriangleXConeT")]
pub struct PyPSDTriangleXConeT {
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub psd_k: usize,
}
#[cfg(feature = "sdp")]
#[pymethods]
impl PyPSDTriangleXConeT {
    #[new]
    pub fn new(indices: Vec<usize>, psd_k: usize) -> PyResult<Self> {
        let expected = psd_k * (psd_k + 1) / 2;
        if indices.len() != expected {
            return Err(PyTypeError::new_err(format!(
                "PSDTriangleXConeT: indices.len() = {} but psd_k = {} requires {}",
                indices.len(),
                psd_k,
                expected
            )));
        }
        Ok(Self { indices, psd_k })
    }
    pub fn __repr__(&self) -> String {
        format!(
            "PSDTriangleXConeT({:?}, psd_k={})",
            self.indices, self.psd_k
        )
    }
}

/// Wrapper for passing direct-x cone specs across the pyo3 boundary,
/// paralleling [`PySupportedCone`]. Extracted from any of the
/// `PyNonnegativeXConeT`, `PySecondOrderXConeT`, or `PyPSDTriangleXConeT`
/// pyclasses.
#[derive(Debug, Clone)]
pub struct PySupportedXCone(pub SupportedXConeT);

impl From<PySupportedXCone> for SupportedXConeT {
    fn from(cone: PySupportedXCone) -> Self {
        cone.0
    }
}

impl<'a> FromPyObject<'a> for PySupportedXCone {
    fn extract_bound(obj: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        let thetype = obj.get_type().name()?;
        let typestr = thetype.to_string_lossy();

        match typestr.as_ref() {
            "NonnegativeXConeT" => {
                let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
                Ok(PySupportedXCone(SupportedXConeT::NonnegativeXConeT(
                    indices,
                )))
            }
            "SecondOrderXConeT" => {
                let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
                Ok(PySupportedXCone(SupportedXConeT::SecondOrderXConeT(
                    indices,
                )))
            }
            "ExponentialXConeT" => {
                let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
                if indices.len() != 3 {
                    return Err(PyTypeError::new_err(format!(
                        "ExponentialXConeT requires 3 indices, got {}",
                        indices.len()
                    )));
                }
                Ok(PySupportedXCone(SupportedXConeT::ExponentialXConeT(
                    indices,
                )))
            }
            "PowerXConeT" => {
                let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
                let alpha: f64 = obj.getattr("alpha")?.extract()?;
                if indices.len() != 3 {
                    return Err(PyTypeError::new_err(format!(
                        "PowerXConeT requires 3 indices, got {}",
                        indices.len()
                    )));
                }
                if !(alpha > 0.0 && alpha < 1.0) {
                    return Err(PyTypeError::new_err(format!(
                        "PowerXConeT alpha must be in (0, 1), got {}",
                        alpha
                    )));
                }
                Ok(PySupportedXCone(SupportedXConeT::PowerXConeT(
                    indices, alpha,
                )))
            }
            "GenPowerXConeT" => {
                let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
                let alphas: Vec<f64> = obj.getattr("alphas")?.extract()?;
                let dim2: usize = obj.getattr("dim2")?.extract()?;
                if indices.len() != alphas.len() + dim2 {
                    return Err(PyTypeError::new_err(format!(
                        "GenPowerXConeT indices.len() = {} but alphas.len() + dim2 = {}",
                        indices.len(),
                        alphas.len() + dim2
                    )));
                }
                Ok(PySupportedXCone(SupportedXConeT::GenPowerXConeT(
                    indices, alphas, dim2,
                )))
            }
            #[cfg(feature = "sdp")]
            "PSDTriangleXConeT" => {
                let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
                let psd_k: usize = obj.getattr("psd_k")?.extract()?;
                Ok(PySupportedXCone(SupportedXConeT::PSDTriangleXConeT(
                    indices, psd_k,
                )))
            }
            _ => {
                let mut errmsg = String::new();
                write!(errmsg, "Unrecognized direct-x cone type : {}", thetype).unwrap();
                Err(PyTypeError::new_err(errmsg))
            }
        }
    }
}

pub(crate) fn _py_to_native_x_cones(x_cones: Vec<PySupportedXCone>) -> Vec<SupportedXConeT> {
    x_cones.into_iter().map(|c| c.into()).collect()
}

// We can't implement the foreign trait FromPyObject directly on
// SupportedCone<f64> since both are defined outside the crate, so
// put a dummy wrapper around it here.

#[derive(Debug, Clone)]
pub struct PySupportedCone(pub SupportedConeT<f64>);

impl From<PySupportedCone> for SupportedConeT<f64> {
    fn from(cone: PySupportedCone) -> Self {
        cone.0
    }
}

impl<'a> FromPyObject<'a> for PySupportedCone {
    fn extract_bound(obj: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        let thetype = obj.get_type().name()?;
        let typestr = thetype.to_string_lossy();

        match typestr.as_ref() {
            "ZeroConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(ZeroConeT(dim)))
            }
            "NonnegativeConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(NonnegativeConeT(dim)))
            }
            "SecondOrderConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(SecondOrderConeT(dim)))
            }
            "ExponentialConeT" => Ok(PySupportedCone(ExponentialConeT())),
            "PowerConeT" => {
                let α: f64 = obj.getattr("α")?.extract()?;
                Ok(PySupportedCone(PowerConeT(α)))
            }
            "GenPowerConeT" => {
                let α: Vec<f64> = obj.getattr("α")?.extract()?;
                let dim2: usize = obj.getattr("dim2")?.extract()?;
                Ok(PySupportedCone(GenPowerConeT(α, dim2)))
            }
            "PSDTriangleConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(PSDTriangleConeT(dim)))
            }
            _ => {
                let mut errmsg = String::new();
                write!(errmsg, "Unrecognized cone type : {}", thetype).unwrap();
                Err(PyTypeError::new_err(errmsg))
            }
        }
    }
}

pub(crate) fn _py_to_native_cones(cones: Vec<PySupportedCone>) -> Vec<SupportedConeT<f64>> {
    //force a vector of PySupportedCone back into a vector
    //of rust native SupportedCone.
    let mut out = Vec::with_capacity(cones.len());
    for cone in cones {
        out.push(cone.into());
    }
    out
}
