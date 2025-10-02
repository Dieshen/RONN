//! Comparative Benchmarks Entry Point
//!
//! Run with: cargo bench --bench comparative --features comparative

mod comparative;

#[cfg(feature = "comparative")]
pub use comparative::vs_onnx_runtime::*;
