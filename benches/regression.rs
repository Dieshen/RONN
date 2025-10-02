//! Regression Benchmarks Entry Point
//!
//! Performance regression tracking against defined targets from TASKS.md.
//!
//! Run with: cargo bench --bench regression

mod regression;

use criterion::criterion_main;

criterion_main!(regression::baseline::regression_benches);
