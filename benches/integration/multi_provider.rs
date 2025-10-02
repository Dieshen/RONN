//! Multi-Provider Integration Benchmarks
//!
//! Compares performance across different execution providers:
//! - CPU Provider
//! - GPU Provider (if available)
//! - BitNet Provider
//! - WASM Provider
//! - Custom Providers
//!
//! Run with: cargo bench --bench integration

use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use ronn_api::{Environment, InferenceSession, SessionConfig};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_providers::ProviderType;
use std::path::PathBuf;
use std::time::Duration;

/// Helper to create test input tensors
fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Benchmark CPU vs GPU vs BitNet providers
pub fn bench_all_providers(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_comparison");

    group.measurement_time(Duration::from_secs(10));

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping provider comparison: test model not found");
        return;
    }

    let providers = vec![
        ("CPU", ProviderType::Cpu),
        ("GPU", ProviderType::Gpu),
        ("BitNet", ProviderType::BitNet),
        ("WASM", ProviderType::Wasm),
    ];

    for (name, provider_type) in providers {
        group.bench_with_input(
            BenchmarkId::new("provider", name),
            &provider_type,
            |b, provider_type| {
                let env = Environment::new("benchmark_env").unwrap();
                let config = SessionConfig {
                    preferred_providers: vec![*provider_type],
                    ..Default::default()
                };

                // Try to create session - skip if provider not available
                let session = match InferenceSession::new(&env, &model_path, config) {
                    Ok(s) => s,
                    Err(_) => {
                        eprintln!("Provider {} not available, skipping", name);
                        return;
                    }
                };

                let input = create_test_input(vec![1, 3, 224, 224]);

                b.iter(|| {
                    let _result = session.run(black_box(vec![input.clone()])).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-GPU scaling (if multiple GPUs available)
pub fn bench_multi_gpu_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_gpu_scaling");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping multi-GPU benchmark: test model not found");
        return;
    }

    // Test with 1, 2, and 4 GPUs if available
    let gpu_counts = [1, 2, 4];

    for gpu_count in gpu_counts.iter() {
        group.bench_with_input(
            BenchmarkId::new("gpu_count", gpu_count),
            gpu_count,
            |b, &gpu_count| {
                let env = Environment::new("benchmark_env").unwrap();
                let config = SessionConfig {
                    preferred_providers: vec![ProviderType::Gpu],
                    num_threads: gpu_count,
                    ..Default::default()
                };

                let session = match InferenceSession::new(&env, &model_path, config) {
                    Ok(s) => s,
                    Err(_) => {
                        eprintln!("Multi-GPU with {} GPUs not available, skipping", gpu_count);
                        return;
                    }
                };

                let input = create_test_input(vec![1, 3, 224, 224]);

                b.iter(|| {
                    let _result = session.run(black_box(vec![input.clone()])).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark provider fallback mechanism
pub fn bench_fallback_mechanism(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_fallback");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping fallback benchmark: test model not found");
        return;
    }

    // Test fallback from GPU to CPU
    group.bench_function("gpu_to_cpu_fallback", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig {
            preferred_providers: vec![ProviderType::Gpu, ProviderType::Cpu],
            ..Default::default()
        };

        let session = InferenceSession::new(&env, &model_path, config).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let _result = session.run(black_box(vec![input.clone()])).unwrap();
        });
    });

    // Test fallback chain: BitNet -> GPU -> CPU
    group.bench_function("full_fallback_chain", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let config = SessionConfig {
            preferred_providers: vec![
                ProviderType::BitNet,
                ProviderType::Gpu,
                ProviderType::Cpu,
            ],
            ..Default::default()
        };

        let session = InferenceSession::new(&env, &model_path, config).unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            let _result = session.run(black_box(vec![input.clone()])).unwrap();
        });
    });

    group.finish();
}

/// Benchmark provider switching overhead
pub fn bench_provider_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_switching");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping provider switching benchmark: test model not found");
        return;
    }

    group.bench_function("switch_cpu_to_gpu", |b| {
        let env = Environment::new("benchmark_env").unwrap();
        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            // Create CPU session
            let cpu_config = SessionConfig {
                preferred_providers: vec![ProviderType::Cpu],
                ..Default::default()
            };
            let cpu_session = InferenceSession::new(&env, &model_path, cpu_config).unwrap();
            let _cpu_result = cpu_session.run(vec![input.clone()]).unwrap();

            // Create GPU session
            let gpu_config = SessionConfig {
                preferred_providers: vec![ProviderType::Gpu],
                ..Default::default()
            };
            if let Ok(gpu_session) = InferenceSession::new(&env, &model_path, gpu_config) {
                let _gpu_result = gpu_session.run(black_box(vec![input.clone()])).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark concurrent execution on multiple providers
pub fn bench_concurrent_providers(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_providers");

    let model_path = PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx");

    if !model_path.exists() {
        eprintln!("Skipping concurrent providers benchmark: test model not found");
        return;
    }

    group.bench_function("cpu_and_gpu_concurrent", |b| {
        let env = Environment::new("benchmark_env").unwrap();

        // Create CPU session
        let cpu_config = SessionConfig {
            preferred_providers: vec![ProviderType::Cpu],
            ..Default::default()
        };
        let cpu_session = InferenceSession::new(&env, &model_path, cpu_config).unwrap();

        // Try to create GPU session
        let gpu_config = SessionConfig {
            preferred_providers: vec![ProviderType::Gpu],
            ..Default::default()
        };
        let gpu_session_opt = InferenceSession::new(&env, &model_path, gpu_config).ok();

        let input = create_test_input(vec![1, 3, 224, 224]);

        b.iter(|| {
            // Run on CPU
            let _cpu_result = cpu_session.run(black_box(vec![input.clone()])).unwrap();

            // Run on GPU if available
            if let Some(ref gpu_session) = gpu_session_opt {
                let _gpu_result = gpu_session.run(black_box(vec![input.clone()])).unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    multi_provider_benches,
    bench_all_providers,
    bench_multi_gpu_scaling,
    bench_fallback_mechanism,
    bench_provider_switching,
    bench_concurrent_providers
);
