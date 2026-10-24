use crate::solver::SolverError;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use rayon::prelude::*;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::collections::HashMap;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Executes independent batch slots using the target's threading support.
pub(crate) struct BatchExecutor {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    pool: Arc<rayon::ThreadPool>,
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
            let pool = shared_thread_pool(threads)?;
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

/// Share rayon workers between live solvers with the same thread count.
/// Only solvers own the pools: after the last solver is dropped, a later
/// constructor must create fresh workers, including after a subsequent fork.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn shared_thread_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>, SolverError> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Weak<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS.get_or_init(Default::default).lock().unwrap();
    if let Some(pool) = pools.get(&num_threads).and_then(Weak::upgrade) {
        return Ok(pool);
    }
    pools.retain(|_, pool| pool.strong_count() > 0);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|_e| SolverError::BadInputData("Failed to configure thread pool"))?,
    );
    pools.insert(num_threads, Arc::downgrade(&pool));
    Ok(pool)
}

#[cfg(all(test, not(all(target_arch = "wasm32", target_os = "unknown"))))]
mod thread_pool_tests {
    use super::*;

    #[test]
    fn live_executors_share_workers_only_for_matching_thread_counts() {
        let first = BatchExecutor::new(2).unwrap();
        let second = BatchExecutor::new(2).unwrap();
        let different = BatchExecutor::new(3).unwrap();
        assert!(Arc::ptr_eq(&first.pool, &second.pool));
        assert!(!Arc::ptr_eq(&first.pool, &different.pool));
        assert_eq!(first.num_threads(), 2);
        assert_eq!(different.num_threads(), 3);
    }
}
