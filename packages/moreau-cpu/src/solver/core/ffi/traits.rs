/// trait for defining FFI data counterparts as associated types
#[allow(missing_docs)]
pub trait MoreauFFI<I> {
    type FFI: From<I>;
}
