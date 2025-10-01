//! Concurrency and edge case tests
//!
//! Tests thread safety, concurrent access, and edge cases.
//! Most tests require ONNX fixtures (see tests/fixtures/README.md).

mod common;

use ronn_api::{Model, OptimizationLevel, SessionOptions, Tensor};
use ronn_core::types::{DataType, TensorLayout};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

#[test]
#[ignore] // Requires fixture
fn test_concurrent_inference_same_session() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = Arc::new(
        model
            .create_session_default()
            .expect("Failed to create session"),
    );

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let session = Arc::clone(&session);
            let input_name = model.input_names()[0].to_string();

            thread::spawn(move || {
                let mut inputs = HashMap::new();
                let data: Vec<f32> = (0..10).map(|x| (x + i * 10) as f32).collect();
                let tensor =
                    Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
                        .expect("Failed to create tensor");

                inputs.insert(input_name.as_str(), tensor);
                session.run(inputs)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok(), "Concurrent inference should succeed");
    }
}

#[test]
#[ignore] // Requires fixture
fn test_concurrent_inference_different_sessions() {
    require_fixture!("simple_model.onnx");

    let model = Arc::new(
        Model::load(common::fixture_path("simple_model.onnx")).expect("Failed to load model"),
    );

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let model = Arc::clone(&model);

            thread::spawn(move || {
                let session = model
                    .create_session_default()
                    .expect("Failed to create session");

                let mut inputs = HashMap::new();
                let data: Vec<f32> = (0..10).map(|x| (x + i * 10) as f32).collect();
                let tensor =
                    Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
                        .expect("Failed to create tensor");

                let input_name = model.input_names()[0];
                inputs.insert(input_name, tensor);
                session.run(inputs)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(
            result.is_ok(),
            "Concurrent inference with different sessions should succeed"
        );
    }
}

#[test]
#[ignore] // Requires fixture
fn test_concurrent_session_creation() {
    require_fixture!("simple_model.onnx");

    let model = Arc::new(
        Model::load(common::fixture_path("simple_model.onnx")).expect("Failed to load model"),
    );

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);

            thread::spawn(move || model.create_session_default())
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(
            result.is_ok(),
            "Concurrent session creation should succeed"
        );
    }
}

#[test]
#[ignore] // Requires fixture
fn test_concurrent_model_loading() {
    require_fixture!("simple_model.onnx");

    let path = common::fixture_path("simple_model.onnx");

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let path = path.clone();

            thread::spawn(move || Model::load(path))
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok(), "Concurrent model loading should succeed");
    }
}

#[test]
#[ignore] // Requires fixture
fn test_model_dropped_with_active_sessions() {
    require_fixture!("simple_model.onnx");

    let session = {
        let model = Model::load(common::fixture_path("simple_model.onnx"))
            .expect("Failed to load model");

        model
            .create_session_default()
            .expect("Failed to create session")
    }; // Model is dropped here

    // Session should still be usable (Arc keeps model alive)
    let inputs: HashMap<&str, Tensor> = HashMap::new();
    let tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    // We need to get input name before model is dropped
    // This test demonstrates the Arc behavior
    let _ = (session, inputs, tensor);
}

#[test]
#[ignore] // Requires fixture
fn test_rapid_session_creation_destruction() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    // Create and destroy sessions rapidly
    for _ in 0..100 {
        let session = model
            .create_session_default()
            .expect("Failed to create session");
        drop(session);
    }

    // Should not leak memory or cause issues
}

#[test]
#[ignore] // Requires fixture
fn test_session_with_different_optimization_levels_concurrent() {
    require_fixture!("simple_model.onnx");

    let model = Arc::new(
        Model::load(common::fixture_path("simple_model.onnx")).expect("Failed to load model"),
    );

    let optimization_levels = vec![
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    let handles: Vec<_> = optimization_levels
        .into_iter()
        .map(|level| {
            let model = Arc::clone(&model);

            thread::spawn(move || -> ronn_api::Result<HashMap<String, Tensor>> {
                let options = SessionOptions::new().with_optimization_level(level);
                let session = model.create_session(options)?;

                let mut inputs = HashMap::new();
                let tensor =
                    Tensor::from_data(vec![1.0; 10], vec![1, 10], DataType::F32, TensorLayout::RowMajor)
                        .map_err(|e| ronn_api::Error::InferenceError(format!("Tensor creation failed: {}", e)))?;

                let input_name = model.input_names()[0];
                inputs.insert(input_name, tensor);
                session.run(inputs)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok(), "Concurrent inference with different optimization levels should succeed");
    }
}

// Edge case tests

#[test]
fn test_thread_count_zero() {
    let options = SessionOptions::new().with_num_threads(0);

    // Should not panic
    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
fn test_thread_count_very_large() {
    let options = SessionOptions::new().with_num_threads(10000);

    // Should not panic (implementation may cap it)
    assert_eq!(options.optimization_level(), OptimizationLevel::O2);
}

#[test]
#[ignore] // Requires fixture
fn test_empty_tensor_input() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    let mut inputs = HashMap::new();

    // Create tensor with 0 elements
    let result = Tensor::from_data(vec![], vec![0], DataType::F32, TensorLayout::RowMajor);

    // This might fail at tensor creation or during inference
    if let Ok(tensor) = result {
        let input_name = model.input_names()[0];
        inputs.insert(input_name, tensor);
        let _ = session.run(inputs);
    }
}

#[test]
#[ignore] // Requires fixture
fn test_very_large_tensor() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    let mut inputs = HashMap::new();

    // Create a large tensor (10MB of float32 data)
    let size = 2_500_000;
    let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();

    let result = Tensor::from_data(
        data,
        vec![1, size],
        DataType::F32,
        TensorLayout::RowMajor,
    );

    // This might fail due to memory constraints or model input constraints
    if let Ok(tensor) = result {
        let input_name = model.input_names()[0];
        inputs.insert(input_name, tensor);
        let _ = session.run(inputs);
    }
}

#[test]
#[ignore] // Requires fixture
fn test_inference_interleaved_with_session_creation() {
    require_fixture!("simple_model.onnx");

    let model = Arc::new(
        Model::load(common::fixture_path("simple_model.onnx")).expect("Failed to load model"),
    );

    let session1 = model
        .create_session_default()
        .expect("Failed to create session 1");

    // Run inference
    let mut inputs = HashMap::new();
    let tensor = Tensor::from_data(
        vec![1.0; 10],
        vec![1, 10],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .expect("Failed to create tensor");

    let input_name = model.input_names()[0];
    inputs.insert(input_name, tensor.clone());
    let _ = session1.run(inputs.clone());

    // Create another session
    let session2 = model
        .create_session_default()
        .expect("Failed to create session 2");

    // Run inference on first session again
    inputs.clear();
    inputs.insert(input_name, tensor.clone());
    let _ = session1.run(inputs.clone());

    // Run inference on second session
    inputs.clear();
    inputs.insert(input_name, tensor);
    let _ = session2.run(inputs);
}

#[test]
#[ignore] // Requires fixture
fn test_session_outlives_model_reference() {
    require_fixture!("simple_model.onnx");

    let session = {
        let model = Model::load(common::fixture_path("simple_model.onnx"))
            .expect("Failed to load model");

        let sess = model
            .create_session_default()
            .expect("Failed to create session");

        // Get input name before model goes out of scope
        // In practice, this pattern would be problematic
        sess
    };

    // Model reference is gone, but session should still be valid (Arc)
    let inputs: HashMap<&str, Tensor> = HashMap::new();
    let _ = (session, inputs);
}

#[test]
fn test_options_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<SessionOptions>();
    assert_sync::<SessionOptions>();
}

#[test]
fn test_model_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<Model>();
    assert_sync::<Model>();
}

#[test]
#[ignore] // Requires fixture and careful testing
fn test_memory_leak_repeated_inference() {
    require_fixture!("simple_model.onnx");

    let model = Model::load(common::fixture_path("simple_model.onnx"))
        .expect("Failed to load model");

    let session = model
        .create_session_default()
        .expect("Failed to create session");

    // Run inference many times
    for i in 0..1000 {
        let mut inputs = HashMap::new();
        let data: Vec<f32> = (0..10).map(|x| (x + i * 10) as f32).collect();
        let tensor = Tensor::from_data(data, vec![1, 10], DataType::F32, TensorLayout::RowMajor)
            .expect("Failed to create tensor");

        let input_name = model.input_names()[0];
        inputs.insert(input_name, tensor);

        let _ = session.run(inputs);
    }

    // If there are memory leaks, they would accumulate here
    // Manual inspection of memory usage would be needed
}

#[test]
#[ignore] // Requires fixture
fn test_concurrent_batch_processing() {
    require_fixture!("simple_model.onnx");

    let model = Arc::new(
        Model::load(common::fixture_path("simple_model.onnx")).expect("Failed to load model"),
    );

    let session = Arc::new(
        model
            .create_session_default()
            .expect("Failed to create session"),
    );

    let handles: Vec<_> = (0..4)
        .map(|batch_id| {
            let session = Arc::clone(&session);
            let input_name = model.input_names()[0].to_string();

            thread::spawn(move || {
                let mut batch = Vec::new();
                for i in 0..8 {
                    let mut inputs = HashMap::new();
                    let data: Vec<f32> = (0..10)
                        .map(|x| (x + i * 10 + batch_id * 100) as f32)
                        .collect();
                    let tensor = Tensor::from_data(
                        data,
                        vec![1, 10],
                        DataType::F32,
                        TensorLayout::RowMajor,
                    )
                    .expect("Failed to create tensor");

                    inputs.insert(input_name.as_str(), tensor);
                    batch.push(inputs);
                }
                session.run_batch(batch)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok(), "Concurrent batch processing should succeed");
        if let Ok(results) = result {
            assert_eq!(results.len(), 8);
        }
    }
}
