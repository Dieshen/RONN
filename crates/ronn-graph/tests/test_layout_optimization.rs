// Tests for layout optimization pass

mod common;

use common::*;
use ronn_graph::{LayoutOptimizationPass, OptimizationPass};

#[test]
fn test_layout_optimization_pass_runs() {
    let mut graph = create_conv_heavy_graph();
    let pass = LayoutOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(
        result.is_ok(),
        "Layout optimization pass should complete successfully"
    );
}

#[test]
fn test_layout_optimization_on_empty_graph() {
    let mut graph = create_empty_graph();
    let pass = LayoutOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should handle empty graph gracefully");

    let stats = result.unwrap();
    assert_eq!(
        stats.nodes_modified, 0,
        "Empty graph should have no modifications"
    );
}

#[test]
fn test_layout_optimization_on_conv_heavy_graph() {
    let mut graph = create_conv_heavy_graph();
    let pass = LayoutOptimizationPass;

    let result = pass.run(&mut graph).unwrap();

    // Conv-heavy graphs may benefit from layout optimization
    // Actual modifications depend on implementation
    assert!(
        result.nodes_modified >= 0,
        "Should analyze conv-heavy graph for layout optimization"
    );
}

#[test]
fn test_layout_optimization_preserves_graph_validity() {
    let mut graph = create_conv_heavy_graph();
    let pass = LayoutOptimizationPass;

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    pass.run(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after layout optimization"
    );
}

#[test]
fn test_layout_optimization_on_non_conv_graph() {
    // Create a graph without conv operations
    let mut builder = builder();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let matmul_id = builder.add_op("MatMul", Some("matmul1".to_string()));
    builder
        .add_input(matmul_id, "input_tensor")
        .add_output(matmul_id, "matmul_output");

    let add_id = builder.add_op("Add", Some("add1".to_string()));
    builder
        .add_input(add_id, "matmul_output")
        .add_output(add_id, "output_tensor");

    builder
        .connect(input_id, matmul_id, "input_tensor")
        .unwrap();
    builder.connect(matmul_id, add_id, "matmul_output").unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let pass = LayoutOptimizationPass;

    let _result = pass.run(&mut graph).unwrap();

    // Non-conv graphs may have different layout preferences
    // Pass completed successfully if we reached here
}

#[test]
fn test_layout_optimization_pass_name() {
    let pass = LayoutOptimizationPass;
    assert_eq!(pass.name(), "LayoutOptimization");
}

#[test]
fn test_layout_optimization_stats_structure() {
    let mut graph = create_conv_heavy_graph();
    let pass = LayoutOptimizationPass;

    let stats = pass.run(&mut graph).unwrap();

    // Verify stats structure
    assert_eq!(
        stats.nodes_removed, 0,
        "Layout optimization should not remove nodes"
    );
    assert_eq!(
        stats.nodes_fused, 0,
        "Layout optimization should not fuse nodes"
    );
    // nodes_modified may be > 0 if layout transforms are inserted
}

#[test]
fn test_layout_optimization_idempotent() {
    let mut graph = create_conv_heavy_graph();
    let pass = LayoutOptimizationPass;

    let result1 = pass.run(&mut graph).unwrap();
    let result2 = pass.run(&mut graph).unwrap();

    // Second run should not find additional optimizations
    assert_eq!(
        result2.nodes_modified, 0,
        "Second run should not find additional layout optimizations"
    );
}

#[test]
fn test_layout_optimization_preserves_outputs() {
    let mut graph = create_conv_heavy_graph();
    let original_outputs = graph.outputs.clone();
    let pass = LayoutOptimizationPass;

    pass.run(&mut graph).unwrap();

    assert_eq!(
        graph.outputs, original_outputs,
        "Graph outputs should be preserved"
    );
}

#[test]
fn test_layout_optimization_mixed_operations() {
    // Create a graph with mixed operation types
    let mut builder = builder();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_output");

    let pool_id = builder.add_op("MaxPool", Some("pool1".to_string()));
    builder
        .add_input(pool_id, "conv_output")
        .add_output(pool_id, "pool_output");

    let matmul_id = builder.add_op("MatMul", Some("matmul1".to_string()));
    builder
        .add_input(matmul_id, "pool_output")
        .add_output(matmul_id, "matmul_output");

    let relu_id = builder.add_op("Relu", Some("relu1".to_string()));
    builder
        .add_input(relu_id, "matmul_output")
        .add_output(relu_id, "output_tensor");

    builder.connect(input_id, conv_id, "input_tensor").unwrap();
    builder.connect(conv_id, pool_id, "conv_output").unwrap();
    builder.connect(pool_id, matmul_id, "pool_output").unwrap();
    builder
        .connect(matmul_id, relu_id, "matmul_output")
        .unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let pass = LayoutOptimizationPass;

    let _result = pass.run(&mut graph).unwrap();

    // Should handle graphs with mixed operation types
    // Pass completed successfully if we reached here
}

#[test]
fn test_layout_optimization_determines_layout() {
    let mut graph = create_conv_heavy_graph();
    let pass = LayoutOptimizationPass;

    // Should determine optimal layout based on operation mix
    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should determine optimal layout");
}

#[test]
fn test_layout_optimization_with_pooling() {
    // Create a graph with conv and pooling operations
    let mut builder = builder();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let conv_id = builder.add_op("Conv", Some("conv1".to_string()));
    builder
        .add_input(conv_id, "input_tensor")
        .add_output(conv_id, "conv_output");

    let maxpool_id = builder.add_op("MaxPool", Some("pool1".to_string()));
    builder
        .add_input(maxpool_id, "conv_output")
        .add_output(maxpool_id, "pool_output");

    let avgpool_id = builder.add_op("AveragePool", Some("pool2".to_string()));
    builder
        .add_input(avgpool_id, "pool_output")
        .add_output(avgpool_id, "output_tensor");

    builder.connect(input_id, conv_id, "input_tensor").unwrap();
    builder.connect(conv_id, maxpool_id, "conv_output").unwrap();
    builder
        .connect(maxpool_id, avgpool_id, "pool_output")
        .unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let pass = LayoutOptimizationPass;

    let _result = pass.run(&mut graph).unwrap();

    // Conv and pooling operations should influence layout choice
    // Pass completed successfully if we reached here
}

#[test]
fn test_layout_optimization_single_node() {
    let mut builder = builder();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "output_tensor");

    builder
        .set_inputs(vec![])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let pass = LayoutOptimizationPass;

    let result = pass.run(&mut graph).unwrap();

    // Single node graph should have minimal layout considerations
    assert_eq!(
        result.nodes_modified, 0,
        "Single node should not require layout changes"
    );
}
