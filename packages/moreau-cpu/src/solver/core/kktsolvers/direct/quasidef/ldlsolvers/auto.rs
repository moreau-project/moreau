#![allow(non_snake_case)]
use crate::{
    algebra::*,
    solver::core::{
        kktsolvers::direct::{BoxedDirectLDLSolver, DirectLDLSolverReqs},
        CoreSettings,
    },
};

#[cfg(not(feature = "faer-sparse"))]
use crate::solver::core::kktsolvers::direct::ldlsolvers::qdldl::QDLDLDirectLDLSolver;

pub struct AutoDirectLDLSolver<T> {
    T: std::marker::PhantomData<T>,
}

impl<T> DirectLDLSolverReqs for AutoDirectLDLSolver<T>
where
    T: FloatT,
{
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}

impl<T> AutoDirectLDLSolver<T>
where
    T: FloatT,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        KKT: &CscMatrix<T>,
        Dsigns: &[i8],
        settings: &CoreSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> BoxedDirectLDLSolver<T> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "faer-sparse")] {
                // Auto mode: always use faer single-threaded.
                // Benchmarks show faer's supernodal LDL^T with Par::Seq
                // is universally faster than QDLDL (simplicial) at all
                // problem sizes, and faster than faer multi-threaded due
                // to rayon overhead at typical KKT dimensions (< 20k).
                use crate::solver::core::kktsolvers::direct::ldlsolvers::faer_ldl::FaerDirectLDLSolver;
                let mut auto_settings = settings.clone();
                auto_settings.ipm.max_threads = 1;
                let solver = FaerDirectLDLSolver::<T>::new(KKT, Dsigns, &auto_settings, perm);
                Box::new(solver)
            } else {
                let solver = QDLDLDirectLDLSolver::<T>::new(KKT, Dsigns, settings, perm);
                Box::new(solver)
            }
        }
    }
}
