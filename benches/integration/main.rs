//! Integration Benchmarks Entry Point
//!
//! Cross-crate and cross-provider performance testing.
//!
//! Run with: cargo bench --bench integration

mod multi_provider;
mod optimization_impact;

use criterion::criterion_main;

criterion_main!(
    multi_provider::multi_provider_benches,
    optimization_impact::optimization_impact_benches
);
