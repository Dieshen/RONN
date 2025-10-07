//! Regression Benchmarks Entry Point
//!
//! Performance regression tracking against defined targets from TASKS.md.
//!
//! Run with: cargo bench --bench regression

mod baseline;

use criterion::criterion_main;

criterion_main!(baseline::regression_benches);
