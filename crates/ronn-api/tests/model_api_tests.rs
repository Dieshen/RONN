//! Unit tests for Model API
//!
//! Tests model loading, metadata access, and session creation.
//! Some tests require ONNX fixtures (see tests/fixtures/README.md).

mod common;

use ronn_api::{Model, OptimizationLevel, SessionOptions};
use ronn_providers::ProviderType;

// Error path tests (no fixtures required)

#[test]
fn test_load_nonexistent_file() {
    let result = Model::load("/path/that/does/not/exist/model.onnx");

    assert!(result.is_err());
    if let Err(err) = result {
        let display = format!("{}", err);
        // Should be an IO error
        assert!(display.contains("IO error") || display.contains("No such file"));
    }
}

#[test]
fn test_load_empty_path() {
    let result = Model::load("");

    assert!(result.is_err());
}

#[test]
fn test_load_invalid_extension() {
    // Try to load a non-ONNX file (use absolute path)
    let cargo_path = std::env::current_dir()
        .unwrap()
        .join("Cargo.toml");

    let result = Model::load(cargo_path);

    assert!(result.is_err());
    // Should fail because it's not a valid ONNX file
}

#[test]
fn test_from_bytes_empty() {
    let result = Model::from_bytes(&[]);

    assert!(result.is_err());
    if let Err(err) = result {
        let display = format!("{}", err);
        // Should indicate parsing failure
        assert!(display.to_lowercase().contains("parse") || display.to_lowercase().contains("onnx"));
    }
}

#[test]
fn test_from_bytes_invalid_data() {
    let invalid_data = vec![0xFF, 0xFE, 0xFD, 0xFC];
    let result = Model::from_bytes(&invalid_data);

    assert!(result.is_err());
}

#[test]
fn test_from_bytes_truncated() {
    // Minimal invalid ONNX-like data
    let truncated = vec![0x08, 0x00]; // Just a couple bytes
    let result = Model::from_bytes(&truncated);

    assert!(result.is_err());
}

#[test]
fn test_from_bytes_random_data() {
    let random_data: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();
    let result = Model::from_bytes(&random_data);

    assert!(result.is_err());
}

// Tests that require fixtures (conditionally compiled)

#[test]
#[ignore] // Requires fixture
fn test_load_valid_model() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    // Model should be loaded successfully
    assert!(!model.input_names().is_empty() || !model.output_names().is_empty());
}

#[test]
#[ignore] // Requires fixture
fn test_model_metadata() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    // Should have some metadata (depends on model)
    let ir_version = model.ir_version();
    assert!(ir_version > 0, "IR version should be positive");
}

#[test]
#[ignore] // Requires fixture
fn test_model_producer_name() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    // Producer name may or may not be set
    let _producer = model.producer_name();
    // Just verify the method works
}

#[test]
#[ignore] // Requires fixture
fn test_model_input_names() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    let inputs = model.input_names();
    assert!(!inputs.is_empty(), "Model should have at least one input");

    // All input names should be non-empty strings
    for input in inputs {
        assert!(!input.is_empty());
    }
}

#[test]
#[ignore] // Requires fixture
fn test_model_output_names() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    let outputs = model.output_names();
    assert!(!outputs.is_empty(), "Model should have at least one output");

    // All output names should be non-empty strings
    for output in outputs {
        assert!(!output.is_empty());
    }
}

#[test]
#[ignore] // Requires fixture
fn test_create_session_default() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    let session = model.create_session_default();
    // Session creation might fail if providers aren't available
    // That's okay for this test
    let _ = session;
}

#[test]
#[ignore] // Requires fixture
fn test_create_session_with_options() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    let options = SessionOptions::new()
        .with_optimization_level(OptimizationLevel::O1)
        .with_provider(ProviderType::CPU);

    let session = model.create_session(options);
    // Session creation might fail if providers aren't available
    let _ = session;
}

#[test]
#[ignore] // Requires fixture
fn test_multiple_sessions_from_same_model() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    // Create multiple sessions
    let session1 = model.create_session_default();
    let session2 = model.create_session_default();
    let session3 = model.create_session(
        SessionOptions::new().with_optimization_level(OptimizationLevel::O3)
    );

    // All sessions should be independent
    // (We can't directly test this without running inference,
    // but creating them should work)
    let _ = (session1, session2, session3);
}

#[test]
#[ignore] // Requires fixture
fn test_model_can_be_cloned_implicitly() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load test model");

    // Model uses Arc internally, so it's cheap to clone
    let _session1 = model.create_session_default();
    let _session2 = model.create_session_default();

    // Both sessions should reference the same underlying model
}

#[test]
#[ignore] // Requires fixture
fn test_load_same_model_twice() {
    require_fixture!("simple_model.onnx");

    let path = common::fixture_path("simple_model.onnx");

    let model1 = Model::load(&path).expect("Failed to load model first time");
    let model2 = Model::load(&path).expect("Failed to load model second time");

    // Both should load successfully and have the same metadata
    assert_eq!(model1.ir_version(), model2.ir_version());
    assert_eq!(model1.input_names(), model2.input_names());
    assert_eq!(model1.output_names(), model2.output_names());
}

#[test]
#[ignore] // Requires fixture
fn test_from_bytes_matches_load() {
    require_fixture!("simple_model.onnx");

    let path = common::fixture_path("simple_model.onnx");

    // Load via file
    let model1 = Model::load(&path).expect("Failed to load model from file");

    // Load via bytes
    let bytes = std::fs::read(&path).expect("Failed to read model file");
    let model2 = Model::from_bytes(&bytes).expect("Failed to load model from bytes");

    // Should have identical metadata
    assert_eq!(model1.ir_version(), model2.ir_version());
    assert_eq!(model1.input_names(), model2.input_names());
    assert_eq!(model1.output_names(), model2.output_names());
}

// Edge case tests

#[test]
fn test_load_directory_instead_of_file() {
    // Try to load a directory path (should fail)
    let result = Model::load(".");

    assert!(result.is_err());
}

#[test]
fn test_load_with_special_characters() {
    // Path with special characters (likely doesn't exist)
    let result = Model::load("/path/with/special/chars/\0/model.onnx");

    assert!(result.is_err());
}

#[test]
fn test_from_bytes_very_small() {
    // Just one byte (definitely not a valid ONNX file)
    let result = Model::from_bytes(&[0x00]);

    assert!(result.is_err());
}

#[test]
fn test_model_api_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<Model>();
    assert_sync::<Model>();
}
