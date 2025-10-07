//! ONNX Model Loading Example
//!
//! This example demonstrates RONN's ONNX model loading and execution capabilities.
//!
//! To generate test models, run:
//! ```bash
//! pip install onnx numpy
//! python scripts/generate_test_model.py
//! ```

use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_onnx::ModelLoader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 RONN ONNX Model Loading Example\n");

    // Example 1: Load ONNX model (if available)
    println!("=== Example 1: ONNX Model Loading ===");
    load_onnx_model_if_available()?;

    // Example 2: Demonstrate model structure
    println!("\n=== Example 2: Model Structure Analysis ===");
    demonstrate_model_structure()?;

    // Example 3: Manual inference demonstration
    println!("\n=== Example 3: Manual Tensor Operations ===");
    demonstrate_manual_inference()?;

    println!("\n✅ All examples completed!");
    println!("\n💡 To generate test ONNX models:");
    println!("   pip install onnx numpy");
    println!("   python scripts/generate_test_model.py");

    Ok(())
}

fn load_onnx_model_if_available() -> Result<(), Box<dyn std::error::Error>> {
    let model_paths = vec![
        "examples/models/simple_add.onnx",
        "examples/models/simple_matmul.onnx",
        "examples/models/simple_relu.onnx",
    ];

    let mut found_model = false;

    for model_path in model_paths {
        if Path::new(model_path).exists() {
            found_model = true;
            println!("📂 Loading model: {}", model_path);

            match ModelLoader::load_from_file(model_path) {
                Ok(loaded_model) => {
                    println!("✅ Successfully loaded model!");
                    println!("   Graph: {}", loaded_model.graph().node_count());
                    println!("   Inputs: {} tensors", loaded_model.inputs().len());
                    println!("   Outputs: {} tensors", loaded_model.outputs().len());
                    println!(
                        "   Initializers: {} weights",
                        loaded_model.initializers().len()
                    );

                    // Display input information
                    for input in loaded_model.inputs() {
                        println!("   Input '{}': shape {:?}", input.name, input.shape);
                    }

                    // Display output information
                    for output in loaded_model.outputs() {
                        println!("   Output '{}': shape {:?}", output.name, output.shape);
                    }

                    println!();
                }
                Err(e) => {
                    println!("❌ Failed to load model: {}", e);
                }
            }
        }
    }

    if !found_model {
        println!("ℹ️  No ONNX models found.");
        println!("   Run 'python scripts/generate_test_model.py' to create test models.");
    }

    Ok(())
}

fn demonstrate_model_structure() -> Result<(), Box<dyn std::error::Error>> {
    println!("ONNX Model Structure:");
    println!("  📊 ModelProto");
    println!("    └─ GraphProto");
    println!("       ├─ Inputs (ValueInfoProto)");
    println!("       ├─ Outputs (ValueInfoProto)");
    println!("       ├─ Nodes (NodeProto)");
    println!("       │  ├─ Op Type (Add, MatMul, ReLU, etc.)");
    println!("       │  ├─ Inputs");
    println!("       │  ├─ Outputs");
    println!("       │  └─ Attributes");
    println!("       └─ Initializers (Weights/Constants)");
    println!();
    println!("Supported Operators (20+):");
    println!("  • Neural Network: Conv2D, BatchNorm, MaxPool, AvgPool");
    println!("  • Activations: ReLU, Sigmoid, Tanh, Softmax, GELU");
    println!("  • Math: Add, Sub, Mul, Div, MatMul");
    println!("  • Tensor: Reshape, Transpose, Concat, Split, Gather, Slice");

    Ok(())
}

fn demonstrate_manual_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating simple Add operation:");

    // Create input tensors (simulating X + Y = Z)
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let y_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let x_tensor = Tensor::from_data(
        x_data.clone(),
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let y_tensor = Tensor::from_data(
        y_data.clone(),
        vec![1, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    println!("  Input X: {:?}", x_data);
    println!("  Input Y: {:?}", y_data);

    // Simulate Add operation (in real implementation, this would use the operator registry)
    // For now, manually compute the result
    let z_data: Vec<f32> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(a, b)| a + b)
        .collect();

    println!("  Output Z: {:?}", z_data);
    println!("  ✅ Operation completed successfully!");

    // Show tensor properties
    println!("\nTensor Properties:");
    println!("  Shape: {:?}", x_tensor.shape());
    println!("  Data Type: {:?}", x_tensor.dtype());
    println!("  Layout: {:?}", x_tensor.layout());
    println!("  Element Count: {}", x_tensor.numel());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_structure_demo() {
        assert!(demonstrate_model_structure().is_ok());
    }

    #[test]
    fn test_manual_inference_demo() {
        assert!(demonstrate_manual_inference().is_ok());
    }

    #[test]
    fn test_tensor_operations() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(
            data.clone(),
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )
        .unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);

        let retrieved = tensor.to_vec().unwrap();
        assert_eq!(retrieved, data);
    }
}
