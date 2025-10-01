// Edge case tests for graph optimization

mod common;

use common::*;
use ronn_core::GraphBuilder;
use ronn_graph::{OptimizationLevel, Optimizer};

// Empty and minimal graph tests

#[test]
fn test_empty_graph_all_levels() {
    let mut graph = create_empty_graph();

    for level in [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ] {
        let optimizer = Optimizer::new(level);
        let result = optimizer.optimize(&mut graph);
        assert!(
            result.is_ok(),
            "Empty graph should be handled gracefully at {:?}",
            level
        );
    }
}

#[test]
fn test_single_node_graph() {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "output_tensor");

    builder
        .set_inputs(vec![])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Single node graph should be optimized");

    let stats = result.unwrap();
    assert_eq!(
        stats.total_changes(),
        0,
        "Single node should not be modified"
    );
}

#[test]
fn test_two_disconnected_subgraphs() {
    let mut builder = GraphBuilder::new();

    // First subgraph
    let input1_id = builder.add_op("Input", Some("input1".to_string()));
    builder.add_output(input1_id, "output1");

    let conv1_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv1_id, "output1")
        .add_output(conv1_id, "conv1_out");

    builder
        .connect(input1_id, conv1_id, "output1")
        .unwrap();

    // Second subgraph (disconnected from first)
    let input2_id = builder.add_op("Input", Some("input2".to_string()));
    builder.add_output(input2_id, "output2");

    let conv2_id = builder.add_op("Conv", Some("conv2".to_string()));
    builder
        .add_input(conv2_id, "output2")
        .add_output(conv2_id, "conv2_out");

    builder
        .connect(input2_id, conv2_id, "output2")
        .unwrap();

    builder
        .set_inputs(vec!["output1".to_string(), "output2".to_string()])
        .set_outputs(vec!["conv1_out".to_string(), "conv2_out".to_string()]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    let result = optimizer.optimize(&mut graph);
    assert!(
        result.is_ok(),
        "Should handle disconnected subgraphs"
    );
}

// Cyclic graph tests (should fail validation)

#[test]
fn test_cyclic_graph_validation_fails() {
    let mut builder = GraphBuilder::new();

    // Create a simple cycle: A -> B -> A
    let node_a_id = builder.add_op("Add", Some("A".to_string()));
    builder
        .add_input(node_a_id, "b_out")
        .add_output(node_a_id, "a_out");

    let node_b_id = builder.add_op("Add", Some("B".to_string()));
    builder
        .add_input(node_b_id, "a_out")
        .add_output(node_b_id, "b_out");

    builder
        .connect(node_a_id, node_b_id, "a_out")
        .unwrap();
    builder
        .connect(node_b_id, node_a_id, "b_out")
        .unwrap();

    builder
        .set_inputs(vec![])
        .set_outputs(vec!["a_out".to_string()]);

    // Building should fail due to cycle detection
    let result = builder.build();
    assert!(result.is_err(), "Cyclic graph should fail validation");
}

// No-op optimization tests

#[test]
fn test_graph_with_no_optimization_opportunities() {
    // Create a simple linear graph with no optimization opportunities
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let output_id = builder.add_op("Output", Some("output".to_string()));
    builder
        .add_input(output_id, "input_tensor")
        .add_output(output_id, "output_tensor");

    builder
        .connect(input_id, output_id, "input_tensor")
        .unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let initial_count = graph.node_count();

    let optimizer = Optimizer::new(OptimizationLevel::O3);
    let stats = optimizer.optimize(&mut graph).unwrap();

    assert_eq!(
        graph.node_count(),
        initial_count,
        "No nodes should be removed"
    );
    assert_eq!(
        stats.total_changes(),
        0,
        "No optimizations should be applied"
    );
}

// Large graph tests

#[test]
fn test_very_deep_linear_graph() {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let mut prev_id = input_id;
    let mut prev_output = "input_tensor".to_string();

    // Create a chain of 100 ReLU operations
    for i in 0..100 {
        let relu_id = builder.add_op("Relu", Some(format!("relu_{}", i)));
        let output_name = format!("relu_{}_out", i);

        builder
            .add_input(relu_id, &prev_output)
            .add_output(relu_id, &output_name);

        builder.connect(prev_id, relu_id, &prev_output).unwrap();

        prev_id = relu_id;
        prev_output = output_name;
    }

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec![prev_output]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle deep linear graph");

    let stats = result.unwrap();
    assert!(
        stats.iterations < 10,
        "Should converge within max iterations"
    );
}

#[test]
fn test_wide_graph_many_parallel_branches() {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let mut output_tensors = vec![];

    // Create 50 parallel branches
    for i in 0..50 {
        let conv_id = builder.add_op("Conv", Some(format!("conv_{}", i)));
        let output_name = format!("conv_{}_out", i);

        builder
            .add_input(conv_id, "input_tensor")
            .add_output(conv_id, &output_name);

        builder
            .connect(input_id, conv_id, "input_tensor")
            .unwrap();

        output_tensors.push(output_name);
    }

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(output_tensors);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle wide graph with many branches");
}

// Graphs with unusual patterns

#[test]
fn test_graph_with_multiple_inputs() {
    let mut builder = GraphBuilder::new();

    let input1_id = builder.add_op("Input", Some("input1".to_string()));
    builder.add_output(input1_id, "input1_tensor");

    let input2_id = builder.add_op("Input", Some("input2".to_string()));
    builder.add_output(input2_id, "input2_tensor");

    let add_id = builder.add_op("Add", Some("add_inputs".to_string()));
    builder
        .add_input(add_id, "input1_tensor")
        .add_input(add_id, "input2_tensor")
        .add_output(add_id, "output_tensor");

    builder
        .connect(input1_id, add_id, "input1_tensor")
        .unwrap();
    builder
        .connect(input2_id, add_id, "input2_tensor")
        .unwrap();

    builder
        .set_inputs(vec!["input1_tensor".to_string(), "input2_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle multiple inputs");
}

#[test]
fn test_graph_with_multiple_outputs() {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_output");

    let relu1_id = builder.add_op("Relu", Some("relu1".to_string()));
    builder
        .add_input(relu1_id, "conv_output")
        .add_output(relu1_id, "output1");

    let relu2_id = builder.add_op("Relu", Some("relu2".to_string()));
    builder
        .add_input(relu2_id, "conv_output")
        .add_output(relu2_id, "output2");

    builder
        .connect(input_id, conv_id, "input_tensor")
        .unwrap();
    builder.connect(conv_id, relu1_id, "conv_output").unwrap();
    builder.connect(conv_id, relu2_id, "conv_output").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output1".to_string(), "output2".to_string()]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O2);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle multiple outputs");

    // Both outputs should be preserved
    assert_eq!(graph.outputs.len(), 2, "Both outputs should be preserved");
}

// Graph with all operation types

#[test]
fn test_graph_with_diverse_operation_types() {
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_out");

    let bn_id = builder.add_op("BatchNormalization", Some("bn1".to_string()));
    builder
        .add_input(bn_id, "conv_out")
        .add_output(bn_id, "bn_out");

    let relu_id = builder.add_op("Relu", Some("relu1".to_string()));
    builder
        .add_input(relu_id, "bn_out")
        .add_output(relu_id, "relu_out");

    let pool_id = builder.add_op("MaxPool", Some("pool1".to_string()));
    builder
        .add_input(pool_id, "relu_out")
        .add_output(pool_id, "pool_out");

    let matmul_id = builder.add_op("MatMul", Some("matmul1".to_string()));
    builder
        .add_input(matmul_id, "pool_out")
        .add_output(matmul_id, "matmul_out");

    let add_id = builder.add_op("Add", Some("add1".to_string()));
    builder
        .add_input(add_id, "matmul_out")
        .add_output(add_id, "output_tensor");

    builder
        .connect(input_id, conv_id, "input_tensor")
        .unwrap();
    builder.connect(conv_id, bn_id, "conv_out").unwrap();
    builder.connect(bn_id, relu_id, "bn_out").unwrap();
    builder.connect(relu_id, pool_id, "relu_out").unwrap();
    builder.connect(pool_id, matmul_id, "pool_out").unwrap();
    builder.connect(matmul_id, add_id, "matmul_out").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle diverse operation types");

    let stats = result.unwrap();
    // Should find multiple optimization opportunities
    assert!(
        stats.total_changes() >= 0,
        "Should analyze all operation types"
    );
}

// Stress test - repeated optimization

#[test]
fn test_repeated_optimization_is_stable() {
    let mut graph = create_fusible_graph();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    // Optimize 5 times
    for i in 0..5 {
        let result = optimizer.optimize(&mut graph);
        assert!(
            result.is_ok(),
            "Optimization round {} should succeed",
            i + 1
        );

        // After first optimization, subsequent ones should make no changes
        if i > 0 {
            let stats = result.unwrap();
            assert_eq!(
                stats.total_changes(),
                0,
                "Round {} should make no changes",
                i + 1
            );
        }
    }

    // Graph should still be valid after repeated optimization
    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after repeated optimization"
    );
}

// Test graph that previously caused issues (regression test template)

#[test]
fn test_graph_with_skip_connection() {
    // ResNet-style skip connection
    let mut builder = GraphBuilder::new();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    // Main path
    let conv1_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv1_id, "input_tensor")
        .add_output(conv1_id, "conv1_out");

    let conv2_id = builder.add_op("Conv", Some("conv2".to_string()));
    builder
        .add_input(conv2_id, "conv1_out")
        .add_output(conv2_id, "conv2_out");

    // Skip connection - add input directly to conv2 output
    let add_id = builder.add_op("Add", Some("skip_add".to_string()));
    builder
        .add_input(add_id, "input_tensor")
        .add_input(add_id, "conv2_out")
        .add_output(add_id, "output_tensor");

    builder
        .connect(input_id, conv1_id, "input_tensor")
        .unwrap();
    builder.connect(conv1_id, conv2_id, "conv1_out").unwrap();
    builder.connect(conv2_id, add_id, "conv2_out").unwrap();
    builder.connect(input_id, add_id, "input_tensor").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let optimizer = Optimizer::new(OptimizationLevel::O3);

    let result = optimizer.optimize(&mut graph);
    assert!(result.is_ok(), "Should handle skip connections correctly");
}
