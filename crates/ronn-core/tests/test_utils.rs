//! Test utilities and helpers for ronn-core tests.
//!
//! This module provides common testing utilities, fixtures, and assertion helpers
//! used across the test suite.

use ronn_core::{DataType, Tensor, TensorLayout};
use anyhow::Result;

/// Create a test tensor with sequential values.
pub fn create_sequential_tensor(shape: Vec<usize>, dtype: DataType) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    Tensor::from_data(data, shape, dtype, TensorLayout::RowMajor)
}

/// Create a test tensor with all ones.
pub fn create_ones_tensor(shape: Vec<usize>, dtype: DataType) -> Result<Tensor> {
    Tensor::ones(shape, dtype, TensorLayout::RowMajor)
}

/// Create a test tensor with all zeros.
pub fn create_zeros_tensor(shape: Vec<usize>, dtype: DataType) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, TensorLayout::RowMajor)
}

/// Create a test tensor with random values.
pub fn create_random_tensor(shape: Vec<usize>, dtype: DataType) -> Result<Tensor> {
    Tensor::rand(shape, dtype, TensorLayout::RowMajor)
}

/// Assert that two tensors are approximately equal within a tolerance.
pub fn assert_tensor_approx_eq(a: &Tensor, b: &Tensor, epsilon: f32) -> Result<()> {
    assert_eq!(a.shape(), b.shape(), "Tensor shapes don't match");

    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;

    for (i, (&a_val, &b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert!(
            (a_val - b_val).abs() < epsilon,
            "Tensors differ at index {}: {} vs {} (diff: {})",
            i, a_val, b_val, (a_val - b_val).abs()
        );
    }

    Ok(())
}

/// Assert that a tensor contains all zeros.
pub fn assert_tensor_all_zeros(tensor: &Tensor) -> Result<()> {
    let data = tensor.to_vec()?;
    for (i, &val) in data.iter().enumerate() {
        assert_eq!(val, 0.0, "Non-zero value at index {}: {}", i, val);
    }
    Ok(())
}

/// Assert that a tensor contains all ones.
pub fn assert_tensor_all_ones(tensor: &Tensor) -> Result<()> {
    let data = tensor.to_vec()?;
    for (i, &val) in data.iter().enumerate() {
        assert!((val - 1.0).abs() < 1e-6, "Non-one value at index {}: {}", i, val);
    }
    Ok(())
}

/// Assert that a tensor contains specific values.
pub fn assert_tensor_eq(tensor: &Tensor, expected: &[f32]) -> Result<()> {
    let data = tensor.to_vec()?;
    assert_eq!(
        data.len(),
        expected.len(),
        "Tensor size mismatch: {} vs {}",
        data.len(),
        expected.len()
    );

    for (i, (&actual, &expected)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "Value mismatch at index {}: {} vs {}",
            i, actual, expected
        );
    }
    Ok(())
}

/// Create a simple test graph for testing purposes.
pub fn create_test_graph() -> Result<ronn_core::ModelGraph> {
    use ronn_core::{AttributeValue, GraphBuilder};

    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input_layer".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv_layer".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_output")
        .add_attribute(conv_id, "kernel_size", AttributeValue::IntArray(vec![3, 3]));

    let relu_id = builder.add_op("ReLU", Some("relu_layer".to_string()));
    builder
        .add_input(relu_id, "conv_output")
        .add_output(relu_id, "relu_output");

    builder.connect(input_id, conv_id, "input_tensor")?;
    builder.connect(conv_id, relu_id, "conv_output")?;

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["relu_output".to_string()]);

    builder.build()
}

/// Create a complex test graph with multiple branches.
pub fn create_complex_test_graph() -> Result<ronn_core::ModelGraph> {
    use ronn_core::{AttributeValue, GraphBuilder};

    let mut builder = GraphBuilder::new();

    // Input layer
    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    // Branch 1: Conv -> ReLU
    let conv1_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv1_id, "input_tensor")
        .add_output(conv1_id, "conv1_out")
        .add_attribute(conv1_id, "kernel_size", AttributeValue::IntArray(vec![3, 3]));

    let relu1_id = builder.add_op("ReLU", Some("relu1".to_string()));
    builder
        .add_input(relu1_id, "conv1_out")
        .add_output(relu1_id, "relu1_out");

    // Branch 2: Conv -> ReLU
    let conv2_id = builder.add_op("Conv", Some("conv2".to_string()));
    builder
        .add_input(conv2_id, "input_tensor")
        .add_output(conv2_id, "conv2_out")
        .add_attribute(conv2_id, "kernel_size", AttributeValue::IntArray(vec![5, 5]));

    let relu2_id = builder.add_op("ReLU", Some("relu2".to_string()));
    builder
        .add_input(relu2_id, "conv2_out")
        .add_output(relu2_id, "relu2_out");

    // Merge: Add
    let add_id = builder.add_op("Add", Some("add".to_string()));
    builder
        .add_input(add_id, "relu1_out")
        .add_input(add_id, "relu2_out")
        .add_output(add_id, "output_tensor");

    // Connect edges
    builder.connect(input_id, conv1_id, "input_tensor")?;
    builder.connect(conv1_id, relu1_id, "conv1_out")?;
    builder.connect(input_id, conv2_id, "input_tensor")?;
    builder.connect(conv2_id, relu2_id, "conv2_out")?;
    builder.connect(relu1_id, add_id, "relu1_out")?;
    builder.connect(relu2_id, add_id, "relu2_out")?;

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    builder.build()
}

/// Benchmark helper to measure operation time.
pub fn measure_time<F, R>(operation: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

/// Generate test shapes for exhaustive testing.
pub fn test_shapes_1d() -> Vec<Vec<usize>> {
    vec![
        vec![1],
        vec![10],
        vec![100],
        vec![1000],
    ]
}

pub fn test_shapes_2d() -> Vec<Vec<usize>> {
    vec![
        vec![1, 1],
        vec![2, 3],
        vec![4, 4],
        vec![10, 20],
        vec![100, 50],
    ]
}

pub fn test_shapes_3d() -> Vec<Vec<usize>> {
    vec![
        vec![1, 1, 1],
        vec![2, 3, 4],
        vec![10, 20, 30],
    ]
}

pub fn test_shapes_4d() -> Vec<Vec<usize>> {
    vec![
        vec![1, 1, 1, 1],
        vec![2, 3, 4, 5],
        vec![1, 3, 224, 224], // Common image size
    ]
}

/// All supported data types for testing.
pub fn test_data_types() -> Vec<DataType> {
    vec![
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::I8,
        DataType::I32,
        DataType::I64,
        DataType::U8,
        DataType::U32,
        DataType::Bool,
        DataType::F64,
    ]
}

/// Common data types for most operations.
pub fn common_data_types() -> Vec<DataType> {
    vec![
        DataType::F32,
        DataType::F16,
        DataType::I32,
    ]
}
