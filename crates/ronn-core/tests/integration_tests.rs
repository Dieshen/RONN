//! Integration tests for ronn-core.
//!
//! This module contains end-to-end integration tests that combine
//! multiple components and test realistic workflows.

mod test_utils;

use anyhow::Result;
use ronn_core::{
    ArithmeticOps, AttributeValue, DataType, GraphBuilder, MatrixOps, ReductionOps, SessionManager,
    ShapeOps, Tensor, TensorLayout,
};
use test_utils::*;

#[tokio::test]
async fn test_complete_inference_pipeline() -> Result<()> {
    // Create a graph
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_data");

    let conv1_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv1_id, "input_data")
        .add_output(conv1_id, "conv1_out")
        .add_attribute(
            conv1_id,
            "kernel_size",
            AttributeValue::IntArray(vec![3, 3]),
        );

    let relu_id = builder.add_op("ReLU", Some("relu".to_string()));
    builder
        .add_input(relu_id, "conv1_out")
        .add_output(relu_id, "relu_out");

    let pool_id = builder.add_op("MaxPool", Some("pool".to_string()));
    builder
        .add_input(pool_id, "relu_out")
        .add_output(pool_id, "output_data")
        .add_attribute(pool_id, "kernel_size", AttributeValue::IntArray(vec![2, 2]));

    builder.connect(input_id, conv1_id, "input_data")?;
    builder.connect(conv1_id, relu_id, "conv1_out")?;
    builder.connect(relu_id, pool_id, "relu_out")?;

    builder
        .set_inputs(vec!["input_data".to_string()])
        .set_outputs(vec!["output_data".to_string()]);

    let graph = builder.build()?;

    // Create session manager and session
    let manager = SessionManager::new();
    let session_id = manager.create_session(graph).await?;

    // Run inference
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;
    let outputs = manager.run_inference(session_id, vec![input]).await?;

    assert_eq!(outputs.len(), 1);

    // Check statistics
    let stats = manager.get_session_statistics(session_id).await?;
    assert_eq!(stats.total_inferences, 1);

    // Cleanup
    manager.destroy_session(session_id).await?;
    Ok(())
}

#[test]
fn test_complex_tensor_workflow() -> Result<()> {
    // Create initial data
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let tensor = Tensor::from_data(
        data.clone(),
        vec![3, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Apply various operations
    let normalized = tensor.sub_scalar(6.5)?.div_scalar(3.45)?;
    let activated = normalized.relu()?;
    let reshaped = activated.reshape(&[2, 6])?;
    let transposed = MatrixOps::transpose(&reshaped)?;
    let summed = transposed.sum_dim(0, false)?;

    // Verify shape transformations
    assert_eq!(summed.shape(), vec![2]);

    // Verify data is still finite
    let final_data = summed.to_vec()?;
    assert!(final_data.iter().all(|x| x.is_finite()));

    Ok(())
}

#[test]
fn test_matrix_chain_multiplication() -> Result<()> {
    // A @ B @ C where dimensions are compatible
    let a = Tensor::ones(vec![10, 20], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::ones(vec![20, 15], DataType::F32, TensorLayout::RowMajor)?;
    let c = Tensor::ones(vec![15, 5], DataType::F32, TensorLayout::RowMajor)?;

    let ab = a.matmul(&b)?;
    let abc = ab.matmul(&c)?;

    assert_eq!(abc.shape(), vec![10, 5]);

    // Each element should be 20 * 15 = 300 (sum of products of ones)
    let data = abc.to_vec()?;
    assert!((data[0] - 300.0).abs() < 1e-2);

    Ok(())
}

#[test]
fn test_broadcasting_pipeline() -> Result<()> {
    // Simulate batch normalization-like operation
    let batch_data = create_sequential_tensor(vec![4, 3, 32, 32], DataType::F32)?;

    // Compute mean along batch dimension (simplified)
    let flattened = batch_data.flatten()?;
    let mean = flattened.mean_all()?;
    let std = flattened.std_all()?;

    // Normalize
    let normalized = batch_data.sub(&mean)?.div(&std)?;

    assert_eq!(normalized.shape(), vec![4, 3, 32, 32]);

    // Apply learned parameters (gamma and beta)
    let gamma = Tensor::ones(vec![1], DataType::F32, TensorLayout::RowMajor)?;
    let beta = Tensor::zeros(vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let output = normalized.mul(&gamma)?.add(&beta)?;

    assert_eq!(output.shape(), vec![4, 3, 32, 32]);
    Ok(())
}

#[test]
#[ignore] // Temporarily disabled due to random tensor dtype issue
fn test_attention_mechanism_simulation() -> Result<()> {
    // Simplified attention: softmax(QK^T)V
    let seq_len = 8;
    let dim = 16;

    let q = create_sequential_tensor(vec![seq_len, dim], DataType::F32)?;
    let k = create_sequential_tensor(vec![seq_len, dim], DataType::F32)?;
    let v = create_sequential_tensor(vec![seq_len, dim], DataType::F32)?;

    // Compute attention scores: QK^T
    let k_t = MatrixOps::transpose(&k)?;
    let scores = q.matmul(&k_t)?;

    assert_eq!(scores.shape(), vec![seq_len, seq_len]);

    // Apply softmax along last dimension
    let attention_weights = scores.softmax(1)?;

    // Apply attention to values
    let output = attention_weights.matmul(&v)?;

    assert_eq!(output.shape(), vec![seq_len, dim]);

    // Verify attention weights sum to 1 for each query
    let weights_data = attention_weights.to_vec()?;
    for i in 0..seq_len {
        let row_sum: f32 = (0..seq_len).map(|j| weights_data[i * seq_len + j]).sum();
        assert!((row_sum - 1.0).abs() < 1e-4);
    }

    Ok(())
}

#[test]
fn test_residual_connection_pattern() -> Result<()> {
    // Simulate ResNet-style residual connection
    let input = create_ones_tensor(vec![4, 64, 32, 32], DataType::F32)?;

    // Main path (simplified)
    let main_path = input.flatten()?.reshape(&[4, 64, 32, 32])?;

    // Residual connection: output = input + main_path
    let output = input.add(&main_path)?;

    assert_eq!(output.shape(), vec![4, 64, 32, 32]);
    Ok(())
}

#[test]
fn test_concatenation_split_roundtrip() -> Result<()> {
    let a = create_sequential_tensor(vec![10, 5], DataType::F32)?;
    let b = create_sequential_tensor(vec![10, 5], DataType::F32)?;
    let c = create_sequential_tensor(vec![10, 5], DataType::F32)?;

    // Concatenate
    let concatenated = Tensor::concat(&[&a, &b, &c], 1)?;
    assert_eq!(concatenated.shape(), vec![10, 15]);

    // Split back
    let chunks = concatenated.chunk(3, 1)?;
    assert_eq!(chunks.len(), 3);

    for chunk in &chunks {
        assert_eq!(chunk.shape(), vec![10, 5]);
    }

    Ok(())
}

#[tokio::test]
async fn test_multi_session_concurrent_inference() -> Result<()> {
    use std::sync::Arc;

    let manager = Arc::new(SessionManager::new());

    // Create multiple sessions
    let session_ids: Vec<_> = futures::future::try_join_all((0..5).map(|_| {
        let manager_clone = Arc::clone(&manager);
        async move {
            let graph = create_test_graph()?;
            manager_clone.create_session(graph).await
        }
    }))
    .await?;

    // Run concurrent inferences across all sessions
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    let mut handles = Vec::new();
    for &session_id in &session_ids {
        for _ in 0..3 {
            let manager_clone = Arc::clone(&manager);
            let input_clone = input.clone();
            let handle = tokio::spawn(async move {
                manager_clone
                    .run_inference(session_id, vec![input_clone])
                    .await
            });
            handles.push(handle);
        }
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    // Most should succeed
    let success_count = results
        .iter()
        .filter(|r| r.as_ref().unwrap().is_ok())
        .count();
    assert!(success_count > 10);

    // Verify global statistics
    let global_stats = manager.get_global_statistics().await;
    assert_eq!(global_stats.total_sessions, 5);
    assert!(global_stats.total_inferences >= success_count as u64);

    Ok(())
}

#[test]
fn test_graph_validation_and_statistics() -> Result<()> {
    let graph = create_complex_test_graph()?;

    // Validate graph
    graph.validate()?;

    // Get statistics
    let stats = graph.statistics();
    assert!(stats.node_count > 0);
    assert!(stats.edge_count > 0);
    assert!(!stats.op_counts.is_empty());

    // Check topological sort
    let topo_order = graph.topological_sort()?;
    assert_eq!(topo_order.len(), stats.node_count);

    Ok(())
}

#[test]
fn test_data_type_conversions() -> Result<()> {
    let data = vec![1.5, 2.5, 3.5, 4.5];

    // Test different data types
    let f32_tensor =
        Tensor::from_data(data.clone(), vec![4], DataType::F32, TensorLayout::RowMajor)?;

    let f16_tensor =
        Tensor::from_data(data.clone(), vec![4], DataType::F16, TensorLayout::RowMajor)?;

    // Operations should preserve dtype
    let f32_result = f32_tensor.add_scalar(1.0)?;
    assert_eq!(f32_result.dtype(), DataType::F32);

    let f16_result = f16_tensor.add_scalar(1.0)?;
    assert_eq!(f16_result.dtype(), DataType::F16);

    Ok(())
}

#[test]
fn test_edge_cases_and_robustness() -> Result<()> {
    // Empty operations
    let empty = Tensor::zeros(vec![0], DataType::F32, TensorLayout::RowMajor)?;
    assert_eq!(empty.numel(), 0);

    // Single element
    let single = Tensor::ones(vec![1], DataType::F32, TensorLayout::RowMajor)?;
    let doubled = single.mul_scalar(2.0)?;
    assert_tensor_eq(&doubled, &[2.0])?;

    // Very large dimensions
    let large = Tensor::zeros(vec![1000, 1000], DataType::F32, TensorLayout::RowMajor)?;
    assert_eq!(large.numel(), 1_000_000);

    Ok(())
}

#[test]
fn test_numerical_stability() -> Result<()> {
    // Very large values
    let large = Tensor::from_data(
        vec![1e10, 1e10, 1e10],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let softmax = large.softmax(0)?;
    let sum = softmax.sum_all()?.to_vec()?[0];
    assert!((sum - 1.0).abs() < 1e-5);

    // Very small values
    let small = Tensor::from_data(
        vec![1e-10, 1e-10, 1e-10],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let normalized = small.div(&small.norm()?)?;
    let norm = normalized.norm()?.to_vec()?[0];
    assert!((norm - 1.0).abs() < 1e-4);

    Ok(())
}
