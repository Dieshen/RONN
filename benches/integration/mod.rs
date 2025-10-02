//! Integration benchmarks module
//!
//! Cross-crate and cross-provider performance testing.

pub mod multi_provider;
pub mod optimization_impact;

pub use multi_provider::*;
pub use optimization_impact::*;
