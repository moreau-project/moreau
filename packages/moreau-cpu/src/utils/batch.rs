use crate::solver::SolverError;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use rayon::prelude::*;

/// Executes independent batch slots using the target's threading support.
pub(crate) struct BatchExecutor {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pool: rayon::ThreadPool,
}

impl BatchExecutor {
    pub(crate) fn new(threads: usize) -> Result<Self, SolverError> {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            if threads > 1 {
                return Err(SolverError::BadInputData(
                    "browser WebAssembly supports only one execution thread",
                ));
            }
            Ok(Self {})
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|_| SolverError::BadInputData("Failed to configure thread pool"))?;
            Ok(Self { pool })
        }
    }

    pub(crate) fn num_threads(&self) -> usize {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            1
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            self.pool.current_num_threads()
        }
    }

    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            f()
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            self.pool.install(f)
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) fn batch_iter<T: Sync>(values: &[T]) -> rayon::slice::Iter<'_, T> {
    values.par_iter()
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) fn batch_iter<T>(values: &[T]) -> std::slice::Iter<'_, T> {
    values.iter()
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) fn batch_range(range: std::ops::Range<usize>) -> rayon::range::Iter<usize> {
    range.into_par_iter()
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) fn batch_range(range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    range
}
