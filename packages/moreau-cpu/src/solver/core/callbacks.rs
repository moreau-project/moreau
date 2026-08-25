// ---------------------------------
// enum for managing callbacks
// ---------------------------------

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

pub(crate) type CallbackFcnFFI<FFI> =
    extern "C" fn(info: *const FFI, data: *mut std::ffi::c_void) -> std::ffi::c_int;
pub trait MoreauCallbackFn<I>: FnMut(&I) -> bool + Send + Sync {}
impl<I, T: FnMut(&I) -> bool + Send + Sync> MoreauCallbackFn<I> for T {}

#[derive(Debug)]
pub(crate) struct CallbackUserDataFFI {
    // AtomicPtr gives us thread-safe pointer storage without a Mutex.
    // The previous `Mutex<*mut c_void>` panicked across the FFI boundary
    // on poison and was unnecessary — the raw pointer is Copy and the
    // pointee's thread-safety is the caller's responsibility (the
    // existing `unsafe impl Send + Sync` for this wrapper was already
    // asserting that). AtomicPtr also auto-derives Send + Sync.
    ptr: AtomicPtr<std::ffi::c_void>,
}

impl CallbackUserDataFFI {
    pub fn new(ptr: *mut std::ffi::c_void) -> Self {
        Self {
            ptr: AtomicPtr::new(ptr),
        }
    }
}

#[derive(Default)]
pub(crate) enum Callback<I, FFI> {
    #[default]
    None,
    Rust(Box<dyn MoreauCallbackFn<I> + Send + Sync>),
    C(CallbackFcnFFI<FFI>, Arc<CallbackUserDataFFI>),
}

impl<I, FFI> std::fmt::Debug for Callback<I, FFI> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Callback::None => write!(f, "Callback::None"),
            Callback::Rust(_) => write!(f, "Callback::Rust(<closure>)"),
            Callback::C(fcn, data) => {
                write!(f, "Callback::C(fcn: {:?}, data: {:?})", fcn, data)
            }
        }
    }
}

impl<I, FFI> Callback<I, FFI>
where
    FFI: From<I>,
    I: Clone + Sized,
{
    pub fn new_c(function: CallbackFcnFFI<FFI>, user_data: *mut std::ffi::c_void) -> Self {
        Callback::C(function, Arc::new(CallbackUserDataFFI::new(user_data)))
    }

    // Call the callback function
    fn call(&mut self, info: &I) -> bool {
        match self {
            Callback::None => false,
            Callback::Rust(ref mut f) => f(info),
            Callback::C(f, data) => {
                let ffi_info = FFI::from(info.clone());
                let rawptr = data.ptr.load(Ordering::Acquire);
                f(&ffi_info as *const FFI, rawptr) != (0 as std::ffi::c_int)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct SolverCallbacks<I, FFI> {
    /// callback for termination
    pub termination_callback: Callback<I, FFI>,
}

impl<I, FFI> Default for SolverCallbacks<I, FFI> {
    // Create a new set of callbacks
    fn default() -> Self {
        Self {
            termination_callback: Callback::None,
        }
    }
}

impl<I, FFI> SolverCallbacks<I, FFI>
where
    FFI: From<I>,
    I: Clone + Sized,
{
    pub(crate) fn check_termination(&mut self, info: &I) -> bool {
        // check termination conditions
        self.termination_callback.call(info)
    }
}
