//! Comparative benchmarks module
//!
//! Compares RONN performance against other ML runtimes.
//! Individual comparison implementations are in submodules.

#[cfg(feature = "comparative")]
pub mod vs_onnx_runtime;

#[cfg(feature = "comparative")]
pub use vs_onnx_runtime::*;

// Placeholder for future comparisons
// pub mod vs_tensorflow_lite;
// pub mod vs_pytorch;
