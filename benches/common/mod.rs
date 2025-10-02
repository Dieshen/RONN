//! Common utilities for benchmarks

use ronn_api::{Model, SessionOptions};
use ronn_core::{DataType, Tensor, TensorLayout};
use ronn_graph::OptimizationLevel;
use ronn_providers::ProviderType;
use std::collections::HashMap;
use std::path::PathBuf;

/// Helper to create test input tensors
pub fn create_test_input(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape, DataType::F32, TensorLayout::RowMajor).unwrap()
}

/// Helper to get model path
pub fn get_model_path() -> PathBuf {
    PathBuf::from("crates/ronn-api/tests/fixtures/simple_model.onnx")
}

/// Helper to create a session with given optimization level
pub fn create_session(opt_level: OptimizationLevel) -> ronn_api::Result<ronn_api::InferenceSession> {
    let model_path = get_model_path();
    let model = Model::load(&model_path)?;
    let options = SessionOptions::new().with_optimization_level(opt_level);
    model.create_session(options)
}

/// Helper to create a session with specific provider
pub fn create_session_with_provider(
    provider: ProviderType,
    opt_level: OptimizationLevel,
) -> ronn_api::Result<ronn_api::InferenceSession> {
    let model_path = get_model_path();
    let model = Model::load(&model_path)?;
    let options = SessionOptions::new()
        .with_optimization_level(opt_level)
        .with_provider(provider);
    model.create_session(options)
}

/// Helper to prepare inputs as HashMap
pub fn prepare_inputs(tensor: Tensor) -> HashMap<&'static str, Tensor> {
    let mut inputs = HashMap::new();
    inputs.insert("input", tensor);
    inputs
}
