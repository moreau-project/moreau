use lazy_static::lazy_static;

lazy_static! {
    /// Cached result of checking MOREAU_DEBUG environment variable.
    /// This avoids repeated env::var calls in hot loops. Gated on
    /// `debug_assertions` so release builds compile away to `false`.
    static ref DEBUG_MODE: bool =
        cfg!(debug_assertions) && std::env::var("MOREAU_DEBUG").is_ok();
}

/// Returns true if MOREAU_DEBUG is set in a debug build. Always `false`
/// in release builds so `debug_println!` and `debug_block!` cost zero.
#[inline]
pub fn is_debug_mode() -> bool {
    *DEBUG_MODE
}

/// Print to stderr when MOREAU_DEBUG (or MOREAU_DEBUG_KKT_DUMP) is set.
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if $crate::utils::debug::is_debug_mode() {
            eprintln!($($arg)*);
        }
    };
}

/// Execute a block of code when MOREAU_DEBUG (or MOREAU_DEBUG_KKT_DUMP) is set.
macro_rules! debug_block {
    ($($body:tt)*) => {
        if $crate::utils::debug::is_debug_mode() {
            $($body)*
        }
    };
}

pub(crate) use debug_block;
pub(crate) use debug_println;
