//! Integration tests for end-to-end workflows
//!
//! Tests complete inference workflows from model loading to result processing.
//! Most tests require ONNX fixtures (see tests/fixtures/README.md).

mod common;

use ronn_api::{Model, OptimizationLevel, SessionOptions, Tensor};
use ronn_core::types::{DataType, TensorLayout};
use ronn_providers::ProviderType;
use std::collections::HashMap;

#[test]
#[ignore] // Requires fixture
fn test_full_inference_workflow() {
    require_fixture!("simple_model.onnx");

    // Load model
    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    // Create session with default options
    let session = model
        .create_session_default()
        .expect("Failed to create session");

    // Prepare inputs (actual tensor shape depends on the model)
    let mut inputs = HashMap::new();
    let input_tensor = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, input_tensor);

    // Run inference
    let outputs = session.run(inputs).expect("Failed to run inference");

    // Verify outputs
    assert!(!outputs.is_empty(), "Should have at least one output");

    for output_name in model.output_names() {
        assert!(
            outputs.contains_key(output_name),
            "Missing expected output: {}",
            output_name
        );
    }
}

#[test]
#[ignore] // Requires fixture
fn test_inference_with_optimization_o0() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O0);

    let session = model
        .create_session(options)
        .expect("Failed to create session");

    let mut inputs = HashMap::new();
    let input_tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, input_tensor);

    let outputs = session.run(inputs).expect("Inference should succeed");
    assert!(!outputs.is_empty());
}

#[test]
#[ignore] // Requires fixture
fn test_inference_with_optimization_o3() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let options = SessionOptions::new().with_optimization_level(OptimizationLevel::O3);

    let session = model
        .create_session(options)
        .expect("Failed to create session");

    let mut inputs = HashMap::new();
    let input_tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, input_tensor);

    let outputs = session.run(inputs).expect("Inference should succeed");
    assert!(!outputs.is_empty());
}

#[test]
#[ignore] // Requires fixture
fn test_missing_input_error() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    // Try to run with no inputs
    let inputs = HashMap::new();
    let result = session.run(inputs);

    // Should fail with InvalidInput error
    assert!(result.is_err());
    let err = result.unwrap_err();
    let display = format!("{}", err);
    assert!(display.contains("input") || display.contains("required"));
}

#[test]
#[ignore] // Requires fixture
fn test_extra_inputs_ignored_or_error() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    let mut inputs = HashMap::new();

    // Add required input
    let input_tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, input_tensor.clone());

    // Add extra input that doesn't exist in model
    inputs.insert("nonexistent_input", input_tensor);

    // This might succeed (extra inputs ignored) or fail (strict validation)
    let _result = session.run(inputs);
    // Behavior depends on implementation
}

#[test]
#[ignore] // Requires fixture
fn test_batch_processing() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    // Create a batch of inputs
    let mut batch = Vec::new();
    for i in 0..3 {
        let mut inputs = HashMap::new();
        let data: Vec<f32> = (0..10).map(|x| (x + i * 10) as f32).collect();
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
            .expect("Failed to create tensor");

        let input_name = model.input_names()[0];
        inputs.insert(input_name, tensor);
        batch.push(inputs);
    }

    // Run batch
    let results = session.run_batch(batch).expect("Batch inference failed");

    assert_eq!(results.len(), 3, "Should have 3 outputs for batch of 3");

    for result in results {
        assert!(!result.is_empty(), "Each result should have outputs");
    }
}

#[test]
#[ignore] // Requires fixture
fn test_multiple_sessions_same_model() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    // Create multiple sessions
    let session1 = model
        .create_session_default()
        .expect("Failed to create session 1");
    let session2 = model
        .create_session_default()
        .expect("Failed to create session 2");
    let session3 = model
        .create_session(SessionOptions::new().with_optimization_level(OptimizationLevel::O1))
        .expect("Failed to create session 3");

    // Prepare inputs
    let mut inputs = HashMap::new();
    let tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, tensor);

    // Run inference on all sessions
    let outputs1 = session1.run(inputs.clone()).expect("Session 1 failed");
    let outputs2 = session2.run(inputs.clone()).expect("Session 2 failed");
    let outputs3 = session3.run(inputs).expect("Session 3 failed");

    // All should succeed
    assert!(!outputs1.is_empty());
    assert!(!outputs2.is_empty());
    assert!(!outputs3.is_empty());
}

#[test]
#[ignore] // Requires fixture
fn test_session_reuse() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    // Run inference multiple times with same session
    for i in 0..5 {
        let mut inputs = HashMap::new();
        let data: Vec<f32> = (0..10).map(|x| (x + i * 10) as f32).collect();
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
            .expect("Failed to create tensor");

        let input_name = model.input_names()[0];
        inputs.insert(input_name, tensor);

        let outputs = session.run(inputs).expect(&format!("Run {} failed", i));
        assert!(!outputs.is_empty());
    }
}

#[test]
#[ignore] // Requires fixture
fn test_inference_with_different_providers() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let providers = vec![ProviderType::CPU, ProviderType::GPU];

    for provider in providers {
        let options = SessionOptions::new().with_provider(provider);

        let session_result = model.create_session(options);

        if let Ok(session) = session_result {
            // Provider is available, try inference
            let mut inputs = HashMap::new();
            let tensor = Tensor::from_data(
                vec![1.0; 10],
                vec![1, 10],
                DataType::F32,
                TensorLayout::RowMajor,
            )
            .expect("Failed to create tensor");

            let input_name = model.input_names()[0];
            inputs.insert(input_name, tensor);

            let outputs = session.run(inputs);
            // Should either succeed or fail gracefully
            let _ = outputs;
        } else {
            // Provider not available, that's okay
            println!("Provider {:?} not available, skipping", provider);
        }
    }
}

#[test]
#[ignore] // Requires fixture
fn test_zero_sized_batch() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    let empty_batch: Vec<HashMap<&str, Tensor>> = Vec::new();
    let results = session.run_batch(empty_batch).expect("Empty batch should succeed");

    assert!(results.is_empty(), "Empty batch should produce empty results");
}

#[test]
#[ignore] // Requires fixture
fn test_large_batch() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    // Create a larger batch
    let mut batch = Vec::new();
    for i in 0..32 {
        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        let data: Vec<f32> = (0..10).map(|x| (x + i * 10) as f32).collect();
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
            .expect("Failed to create tensor");

        let input_name = model.input_names()[0];
        inputs.insert(input_name, tensor);
        batch.push(inputs);
    }

    let results = session.run_batch(batch).expect("Large batch failed");
    assert_eq!(results.len(), 32);
}

#[test]
#[ignore] // Requires fixture
fn test_session_options_preserved() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O3)
        .with_provider(ProviderType::CPU);

    let session = model
        .create_session(options.clone())
        .expect("Failed to create session");

    // Verify options are preserved
    assert_eq!(
        session.options().optimization_level(),
        OptimizationLevel::O3
    );
    assert_eq!(session.options().provider_type(), ProviderType::CPU);
}

#[test]
#[ignore] // Requires fixture
fn test_model_from_bytes_inference() {
    require_fixture!("simple_model.onnx");

    let path = common::fixture_path("simple_model.onnx");
    let bytes = std::fs::read(&path).expect("Failed to read model file");

    let model = Model::from_bytes(&bytes).expect("Failed to load from bytes");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    let mut inputs = HashMap::new();
    let tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, tensor);

    let outputs = session.run(inputs).expect("Inference failed");
    assert!(!outputs.is_empty());
}

// Note: Async inference test would require #[tokio::test]
// Currently skipped as async implementation is not complete
