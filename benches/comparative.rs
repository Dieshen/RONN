//! Comparative Benchmarks Entry Point
//!
//! Run with: cargo bench --bench comparative --features comparative

mod comparative_impl;

#[cfg(feature = "comparative")]
pub use comparative_impl::vs_onnx_runtime::*;
