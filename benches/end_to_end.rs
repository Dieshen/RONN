//! End-to-End Benchmarks for RONN Runtime
//!
//! Comprehensive benchmarks covering the full inference pipeline:
//! - Model loading and initialization
//! - Graph optimization at different levels
//! - Inference execution with different providers
//! - Memory usage and throughput measurements
//!
//! Run with: cargo bench --bench end_to_end

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ronn_api::{Environment, InferenceSession, SessionConfig};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_graph::OptimizationLevel;
use std::path::PathBuf;
use std::time::Duration;

/// Helper to create test input tensors
fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Benchmark full pipeline: Load → Optimize → Execute
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    // Configure measurement parameters
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Get path to test model
    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping full_pipeline benchmark: test model not found at {:?}", model_path);
        return;
    }

    // Test different optimization levels
    let optimization_levels = [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    for opt_level in optimization_levels.iter() {
        group.bench_with_input(
            BenchmarkId::new("optimization_level", format!("{:?}", opt_level)),
            opt_level,
            |b, &opt_level| {
                b.iter(|| {
                    // Create environment and session
                    let env = Environment::new("benchmark_env").unwrap();
                    let config = SessionConfig {
                        optimization_level: opt_level,
                        ..Default::default()
                    };

                    // Load and optimize model
                    let session = InferenceSession::new(
                        &env,
                        black_box(&model_path),
                        black_box(config),
                    ).unwrap();

                    // Create input
                    let input = create_test_input(vec![1, 3, 224, 224]);

                    // Run inference
                    let _result = session.run(black_box(vec![input])).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark model loading time
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping model_loading benchmark: test model not found");
        return;
    }

    group.bench_function("load_simple_model", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig::default();

        b.iter(|| {
            let _session = InferenceSession::new(
                &env,
                black_box(&model_path),
                black_box(config.clone()),
            ).unwrap();
        });
    });

    group.finish();
}

/// Benchmark inference latency with different batch sizes
fn bench_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_latency");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping inference_latency benchmark: test model not found");
        return;
    }

    let env = Environment::new("benchmark_env").unwrap();
    let config = SessionConfig {
        optimization_level: OptimizationLevel::O2,
        ..Default::default()
    };

    let session = InferenceSession::new(&env, &model_path, config).unwrap();

    // Test different batch sizes
    let batch_sizes = [1, 4, 8, 16, 32];

    for batch_size in batch_sizes.iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                let input = create_test_input(vec![batch_size, 3, 224, 224]);

                b.iter(|| {
                    let _result = session.run(black_box(vec![input.clone()])).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark inference throughput (inferences per second)
fn bench_inference_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_throughput");

    // Increase measurement time for throughput tests
    group.measurement_time(Duration::from_secs(15));

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping inference_throughput benchmark: test model not found");
        return;
    }

    let env = Environment::new("benchmark_env").unwrap();
    let config = SessionConfig {
        optimization_level: OptimizationLevel::O3,
        ..Default::default()
    };

    let session = InferenceSession::new(&env, &model_path, config).unwrap();
    let input = create_test_input(vec![1, 3, 224, 224]);

    group.bench_function("continuous_inference", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let _result = session.run(black_box(vec![input.clone()])).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping memory_usage benchmark: test model not found");
        return;
    }

    // Test different tensor sizes to measure memory allocation patterns
    let tensor_sizes = [
        (1, 3, 64, 64),      // Small
        (1, 3, 224, 224),    // Medium
        (1, 3, 512, 512),    // Large
    ];

    for (i, &(batch, channels, height, width)) in tensor_sizes.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("tensor_size", i),
            &(batch, channels, height, width),
            |b, &(batch, channels, height, width)| {
                let env = Environment::new("benchmark_env").unwrap();
                let config = SessionConfig::default();
                let session = InferenceSession::new(&env, &model_path, config).unwrap();

                b.iter(|| {
                    let input = create_test_input(vec![batch, channels, height, width]);
                    let _result = session.run(black_box(vec![input])).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cold start vs warm start performance
fn bench_cold_vs_warm_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_vs_warm");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping cold_vs_warm benchmark: test model not found");
        return;
    }

    // Cold start: create new session each time
    group.bench_function("cold_start", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let config = SessionConfig::default();
            let session = InferenceSession::new(&env, black_box(&model_path), config).unwrap();
            let _result = session.run(black_box(vec![input.clone()])).unwrap();
        });
    });

    // Warm start: reuse session
    group.bench_function("warm_start", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig::default();
        let session = InferenceSession::new(&env, &model_path, config).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let _result = session.run(black_box(vec![input.clone()])).unwrap();
        });
    });

    group.finish();
}

/// Benchmark optimization pass overhead
fn bench_optimization_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_overhead");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping optimization_overhead benchmark: test model not found");
        return;
    }

    let optimization_levels = [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    let env = Environment::new("benchmark_env").unwrap();

    for opt_level in optimization_levels.iter() {
        group.bench_with_input(
            BenchmarkId::new("opt_level", format!("{:?}", opt_level)),
            opt_level,
            |b, &opt_level| {
                b.iter(|| {
                    let config = SessionConfig {
                        optimization_level: opt_level,
                        ..Default::default()
                    };
                    let _session = InferenceSession::new(
                        &env,
                        black_box(&model_path),
                        black_box(config),
                    ).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    end_to_end_benches,
    bench_full_pipeline,
    bench_model_loading,
    bench_inference_latency,
    bench_inference_throughput,
    bench_memory_usage,
    bench_cold_vs_warm_start,
    bench_optimization_overhead
);

criterion_main!(end_to_end_benches);
