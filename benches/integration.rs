//! Integration Benchmarks Entry Point
//!
//! Cross-crate and cross-provider performance testing.
//!
//! Run with: cargo bench --bench integration

mod integration;

use criterion::criterion_main;

criterion_main!(
    integration::multi_provider::multi_provider_benches,
    integration::optimization_impact::optimization_impact_benches
);
