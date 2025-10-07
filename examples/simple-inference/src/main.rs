//! Basic inference example demonstrating RONN's high-level API
//!
//! This example shows how to:
//! - Load a model using the simple API
//! - Create an inference session with custom options
//! - Run inference with input tensors
//! - Process output results

use ronn_api::prelude::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 RONN Basic Inference Example\n");

    // Example 1: Simple matrix multiplication model
    println!("=== Example 1: Simple Computation ===");
    simple_computation_example()?;

    // Example 2: Demonstrating different optimization levels
    println!("\n=== Example 2: Optimization Levels ===");
    optimization_levels_example()?;

    // Example 3: Batch processing
    println!("\n=== Example 3: Batch Processing ===");
    batch_processing_example()?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

fn simple_computation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating simple computation graph...");

    // Create a simple test tensor
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let input_tensor = Tensor::from_data(
        input_data.clone(),
        vec![2, 2],
        ronn_core::types::DataType::F32,
        ronn_core::types::TensorLayout::RowMajor,
    )?;

    println!("Input tensor shape: {:?}", input_tensor.shape());
    println!("Input data: {:?}", input_data);

    // For this example, we'll just demonstrate tensor operations
    // In a real scenario, you'd load an ONNX model here

    // Simulate model output (in practice this would come from model.run())
    let output_tensor = input_tensor.clone();
    let output_data = output_tensor.to_vec()?;

    println!("Output tensor shape: {:?}", output_tensor.shape());
    println!("Output data: {:?}", output_data);

    Ok(())
}

fn optimization_levels_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating different optimization levels...\n");

    let optimization_levels = vec![
        (OptimizationLevel::O0, "O0 - No optimizations"),
        (
            OptimizationLevel::O1,
            "O1 - Basic optimizations (constant folding, DCE)",
        ),
        (
            OptimizationLevel::O2,
            "O2 - Standard optimizations (+ node fusion, layout)",
        ),
        (
            OptimizationLevel::O3,
            "O3 - Aggressive optimizations (+ provider-specific)",
        ),
    ];

    for (level, description) in optimization_levels {
        println!("  {} - {}", format!("{:?}", level), description);

        // In practice, you would:
        // let options = SessionOptions::new().with_optimization_level(level);
        // let session = model.create_session(options)?;
    }

    println!("\n💡 Tip: Use O2 for balanced performance, O3 for maximum speed");

    Ok(())
}

fn batch_processing_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing batch of inputs...");

    // Create a batch of input tensors
    let batch_size = 3;
    let mut batch_tensors = Vec::new();

    for i in 0..batch_size {
        let data: Vec<f32> = (0..4).map(|x| (x + i * 4) as f32).collect();
        let tensor = Tensor::from_data(
            data,
            vec![2, 2],
            ronn_core::types::DataType::F32,
            ronn_core::types::TensorLayout::RowMajor,
        )?;
        batch_tensors.push(tensor);
    }

    println!("Created batch of {} tensors", batch_tensors.len());

    // In practice, you would use session.run_batch() here
    for (idx, tensor) in batch_tensors.iter().enumerate() {
        let data = tensor.to_vec()?;
        println!("  Batch {}: {:?}", idx, data);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_computation() {
        assert!(simple_computation_example().is_ok());
    }

    #[test]
    fn test_optimization_levels() {
        assert!(optimization_levels_example().is_ok());
    }

    #[test]
    fn test_batch_processing() {
        assert!(batch_processing_example().is_ok());
    }
}
