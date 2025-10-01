//! Unit tests for SessionOptions configuration
//!
//! Tests the builder pattern, default values, and configuration options
//! for inference sessions.

use ronn_api::{OptimizationLevel, SessionOptions};
use ronn_providers::ProviderType;

#[test]
fn test_default_options() {
    let options = SessionOptions::default();

    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
    assert_eq!(options.provider_type(), ProviderType::CPU);
}

#[test]
fn test_new_creates_default() {
    let options1 = SessionOptions::new();
    let options2 = SessionOptions::default();

    assert_eq!(options1.optimization_level(), options2.optimization_level());
    assert_eq!(options1.provider_type(), options2.provider_type());
}

#[test]
fn test_with_optimization_level_o0() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O0);

    assert_eq!(options.optimization_level(), OptimizationLevel::O0);
}

#[test]
fn test_with_optimization_level_o1() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O1);

    assert_eq!(options.optimization_level(), OptimizationLevel::O1);
}

#[test]
fn test_with_optimization_level_o2() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O2);

    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_with_optimization_level_o3() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O3);

    assert_eq!(options.optimization_level(), OptimizationLevel::O3);
}

#[test]
fn test_with_provider_cpu() {
    let options = SessionOptions::new()
        .with_provider(ProviderType::CPU);

    assert_eq!(options.provider_type(), ProviderType::CPU);
}

#[test]
fn test_with_provider_gpu() {
    let options = SessionOptions::new()
        .with_provider(ProviderType::GPU);

    assert_eq!(options.provider_type(), ProviderType::GPU);
}

#[test]
fn test_with_num_threads() {
    let options = SessionOptions::new()
        .with_num_threads(4);

    // Note: num_threads is private, so we can't directly assert it
    // This test verifies the builder method compiles and returns Self
    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_with_num_threads_single() {
    let options = SessionOptions::new()
        .with_num_threads(1);

    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_with_num_threads_many() {
    let options = SessionOptions::new()
        .with_num_threads(16);

    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_with_profiling_enabled() {
    let options = SessionOptions::new()
        .with_profiling(true);

    // Profiling field is private, so we verify the builder works
    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_with_profiling_disabled() {
    let options = SessionOptions::new()
        .with_profiling(false);

    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_builder_chain_all_options() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O3)
        .with_provider(ProviderType::GPU)
        .with_num_threads(8)
        .with_profiling(true);

    assert_eq!(options.optimization_level(), OptimizationLevel::O3);
    assert_eq!(options.provider_type(), ProviderType::GPU);
}

#[test]
fn test_builder_chain_partial() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O1)
        .with_provider(ProviderType::CPU);

    assert_eq!(options.optimization_level(), OptimizationLevel::O1);
    assert_eq!(options.provider_type(), ProviderType::CPU);
}

#[test]
fn test_builder_overwrite_optimization_level() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O0)
        .with_optimization_level(OptimizationLevel::O3);

    // Last call should win
    assert_eq!(options.optimization_level(), OptimizationLevel::O3);
}

#[test]
fn test_builder_overwrite_provider() {
    let options = SessionOptions::new()
        .with_provider(ProviderType::GPU)
        .with_provider(ProviderType::CPU);

    // Last call should win
    assert_eq!(options.provider_type(), ProviderType::CPU);
}

#[test]
fn test_options_clone() {
    let options1 = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O3)
        .with_provider(ProviderType::GPU);

    let options2 = options1.clone();

    assert_eq!(options1.optimization_level(), options2.optimization_level());
    assert_eq!(options1.provider_type(), options2.provider_type());
}

#[test]
fn test_options_debug_format() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O2)
        .with_provider(ProviderType::CPU);

    let debug_str = format!("{:?}", options);

    // Should contain the struct name and key fields
    assert!(debug_str.contains("SessionOptions"));
}

#[test]
fn test_multiple_independent_options() {
    let opts1 = SessionOptions::new().with_optimization_level(OptimizationLevel::O0);
    let opts2 = SessionOptions::new().with_optimization_level(OptimizationLevel::O3);
    let opts3 = SessionOptions::new().with_provider(ProviderType::GPU);

    // Each should be independent
    assert_eq!(opts1.optimization_level(), OptimizationLevel::O0);
    assert_eq!(opts2.optimization_level(), OptimizationLevel::O3);
    assert_eq!(opts3.provider_type(), ProviderType::GPU);
    assert_eq!(opts3.optimization_level(), OptimizationLevel::O2); // default
}

#[test]
fn test_options_move_semantics() {
    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O3);

    // Move the options
    let moved_options = options;

    // Can still use moved_options
    assert_eq!(moved_options.optimization_level(), OptimizationLevel::O3);
}

#[test]
fn test_all_optimization_levels_distinct() {
    let o0 = OptimizationLevel::O0;
    let o1 = OptimizationLevel::O1;
    let o2 = OptimizationLevel::O2;
    let o3 = OptimizationLevel::O3;

    // Each level should be distinct
    assert_ne!(o0, o1);
    assert_ne!(o1, o2);
    assert_ne!(o2, o3);
    assert_ne!(o0, o3);
}

#[test]
fn test_provider_types_distinct() {
    let cpu = ProviderType::CPU;
    let gpu = ProviderType::GPU;

    assert_ne!(cpu, gpu);
}

#[test]
fn test_session_options_reusable() {
    let base_options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O2);

    // Clone and modify
    let gpu_options = base_options.clone()
        .with_provider(ProviderType::GPU);

    let cpu_options = base_options.clone()
        .with_provider(ProviderType::CPU);

    // Original should be unchanged
    assert_eq!(base_options.optimization_level(), OptimizationLevel::O2);

    // Modified versions should have different providers
    assert_eq!(gpu_options.provider_type(), ProviderType::GPU);
    assert_eq!(cpu_options.provider_type(), ProviderType::CPU);
}
