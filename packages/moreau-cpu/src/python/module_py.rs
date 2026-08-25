use super::*;
use pyo3::prelude::*;

#[pyfunction(name = "force_load_blas_lapack")]
fn force_load_blas_lapack_py(py: Python<'_>) -> PyResult<()> {
    //force BLAS/LAPACK fcn pointer load
    //when using scipy lapack/blas
    #[cfg(sdp_pyblas)]
    {
        // Pre-validate so a missing scipy install surfaces as a clean
        // ImportError on `import moreau`. The downstream lazy_static in
        // pyblas/blas_wrappers.rs uses .expect(), which under
        // panic="abort" would otherwise abort the whole interpreter.
        // (#189)
        crate::python::pyblas::PyBlasPointers::new(py).map_err(|e| {
            pyo3::exceptions::PyImportError::new_err(format!(
                "moreau requires scipy for BLAS/LAPACK bindings; failed to load BLAS: {}",
                e
            ))
        })?;
        crate::python::pyblas::PyLapackPointers::new(py).map_err(|e| {
            pyo3::exceptions::PyImportError::new_err(format!(
                "moreau requires scipy for BLAS/LAPACK bindings; failed to load LAPACK: {}",
                e
            ))
        })?;
        crate::python::pyblas::force_load();
        let _ = py;
    }
    Ok(())
}

// get/set for the solver's internal infinity limit
#[pyfunction(name = "get_infinity")]
fn get_infinity_py() -> f64 {
    crate::get_infinity()
}
#[pyfunction(name = "set_infinity")]
fn set_infinity_py(v: f64) {
    crate::set_infinity(v);
}
#[pyfunction(name = "default_infinity")]
fn default_infinity_py() {
    crate::default_infinity();
}
#[pyfunction(name = "buildinfo")]
fn buildinfo_py() {
    crate::buildinfo();
}

// Python module and registry, which includes registration of the
// data types defined in the other files in this rust module
#[pymodule]
fn _cpu_solver(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    //module version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // module initializer, called on module import
    m.add_function(wrap_pyfunction!(force_load_blas_lapack_py, m)?)
        .unwrap();

    //module globs
    m.add_function(wrap_pyfunction!(get_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(set_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(default_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(buildinfo_py, m)?).unwrap();

    // API Cone types
    m.add_class::<PyZeroConeT>()?;
    m.add_class::<PyNonnegativeConeT>()?;
    m.add_class::<PySecondOrderConeT>()?;
    m.add_class::<PyExponentialConeT>()?;
    m.add_class::<PyPowerConeT>()?;
    m.add_class::<PyGenPowerConeT>()?;
    m.add_class::<PyPSDTriangleConeT>()?;
    m.add_class::<super::cones_py::PyNonnegativeXConeT>()?;
    m.add_class::<super::cones_py::PySecondOrderXConeT>()?;
    m.add_class::<super::cones_py::PyExponentialXConeT>()?;
    m.add_class::<super::cones_py::PyPowerXConeT>()?;
    m.add_class::<super::cones_py::PyGenPowerXConeT>()?;
    #[cfg(feature = "sdp")]
    m.add_class::<super::cones_py::PyPSDTriangleXConeT>()?;

    //other API data types
    m.add_class::<PySolverStatus>()?;
    m.add_class::<PySolverType>()?;
    m.add_class::<PyDefaultSolution>()?;
    m.add_class::<PyBackwardResult>()?;
    m.add_class::<PyIPMSettings>()?;
    m.add_class::<PyDefaultSettings>()?;
    m.add_class::<PyDefaultInfo>()?;
    m.add_class::<PyLinearSolverInfo>()?;

    // Main solver object
    m.add_class::<PyDefaultSolver>()?;

    // Batch processing types
    m.add_class::<PyBatchProblem>()?;
    m.add_class::<PyUpstreamGradients>()?;
    m.add_class::<PyComputedGradients>()?;
    m.add_class::<PyCompiledSolver>()?;
    m.add_class::<PyBatchedSolutionFlat>()?;
    m.add_class::<PyBatchedGradientsFlat>()?;

    // Active-set solver (only when compiled with active-set feature)
    #[cfg(feature = "active-set")]
    {
        m.add_class::<super::active_set_py::PyActiveSetSettings>()?;
        m.add_class::<super::active_set_py::PyActiveSetBackwardState>()?;
        m.add_class::<super::active_set_py::PyActiveSetSolver>()?;
    }

    Ok(())
}
