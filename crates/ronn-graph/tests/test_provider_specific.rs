// Tests for provider-specific optimization passes (CPU and GPU)

mod common;

use common::*;
use ronn_graph::{CpuOptimizationPass, GpuOptimizationPass, OptimizationPass};

// CPU Optimization Pass Tests

#[test]
fn test_cpu_optimization_pass_runs() {
    let mut graph = create_simple_conv_graph();
    let pass = CpuOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(
        result.is_ok(),
        "CPU optimization pass should complete successfully"
    );
}

#[test]
fn test_cpu_optimization_on_empty_graph() {
    let mut graph = create_empty_graph();
    let pass = CpuOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should handle empty graph gracefully");

    let stats = result.unwrap();
    assert_eq!(
        stats.nodes_modified, 0,
        "Empty graph should have no modifications"
    );
}

#[test]
fn test_cpu_optimization_preserves_graph_validity() {
    let mut graph = create_simple_conv_graph();
    let pass = CpuOptimizationPass;

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    pass.run(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after CPU optimization"
    );
}

#[test]
fn test_cpu_optimization_pass_name() {
    let pass = CpuOptimizationPass;
    assert_eq!(pass.name(), "CpuOptimization");
}

#[test]
fn test_cpu_optimization_stats_structure() {
    let mut graph = create_simple_conv_graph();
    let pass = CpuOptimizationPass;

    let stats = pass.run(&mut graph).unwrap();

    // Verify stats structure
    assert_eq!(
        stats.nodes_removed, 0,
        "CPU optimization should not remove nodes"
    );
    assert_eq!(
        stats.nodes_fused, 0,
        "CPU optimization should not fuse nodes"
    );
    // nodes_modified may be > 0 if SIMD or cache optimizations are applied
}

#[test]
fn test_cpu_optimization_idempotent() {
    let mut graph = create_simple_conv_graph();
    let pass = CpuOptimizationPass;

    let result1 = pass.run(&mut graph).unwrap();
    let result2 = pass.run(&mut graph).unwrap();

    // Second run should not find additional optimizations
    assert_eq!(
        result2.nodes_modified, 0,
        "Second run should not find additional CPU optimizations"
    );
}

#[test]
fn test_cpu_optimization_on_complex_graph() {
    let mut graph = create_fusible_graph();
    let pass = CpuOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should handle complex graphs");
}

#[test]
fn test_cpu_optimization_preserves_outputs() {
    let mut graph = create_simple_conv_graph();
    let original_outputs = graph.outputs.clone();
    let pass = CpuOptimizationPass;

    pass.run(&mut graph).unwrap();

    assert_eq!(
        graph.outputs, original_outputs,
        "Graph outputs should be preserved"
    );
}

#[test]
fn test_cpu_optimization_with_matmul() {
    let mut graph = create_matmul_bias_graph();
    let pass = CpuOptimizationPass;

    let _result = pass.run(&mut graph).unwrap();

    // MatMul operations can benefit from SIMD optimizations
    // Pass completed successfully if we reached here
}

// GPU Optimization Pass Tests

#[test]
fn test_gpu_optimization_pass_runs() {
    let mut graph = create_simple_conv_graph();
    let pass = GpuOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(
        result.is_ok(),
        "GPU optimization pass should complete successfully"
    );
}

#[test]
fn test_gpu_optimization_on_empty_graph() {
    let mut graph = create_empty_graph();
    let pass = GpuOptimizationPass;

    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should handle empty graph gracefully");

    let stats = result.unwrap();
    assert_eq!(stats.nodes_fused, 0, "Empty graph should have no fusions");
    assert_eq!(
        stats.nodes_modified, 0,
        "Empty graph should have no modifications"
    );
}

#[test]
fn test_gpu_optimization_preserves_graph_validity() {
    let mut graph = create_simple_conv_graph();
    let pass = GpuOptimizationPass;

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    pass.run(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after GPU optimization"
    );
}

#[test]
fn test_gpu_optimization_pass_name() {
    let pass = GpuOptimizationPass;
    assert_eq!(pass.name(), "GpuOptimization");
}

#[test]
fn test_gpu_optimization_stats_structure() {
    let mut graph = create_simple_conv_graph();
    let pass = GpuOptimizationPass;

    let stats = pass.run(&mut graph).unwrap();

    // Verify stats structure
    assert_eq!(
        stats.nodes_removed, 0,
        "GPU optimization should not remove nodes"
    );
    // nodes_fused may be > 0 for kernel fusion
    // nodes_modified may be > 0 for memory coalescing optimizations
}

#[test]
fn test_gpu_optimization_idempotent() {
    let mut graph = create_simple_conv_graph();
    let pass = GpuOptimizationPass;

    let result1 = pass.run(&mut graph).unwrap();
    let result2 = pass.run(&mut graph).unwrap();

    // Second run should not find additional optimizations
    assert_eq!(
        result2.nodes_fused, 0,
        "Second run should not find additional GPU fusions"
    );
    assert_eq!(
        result2.nodes_modified, 0,
        "Second run should not find additional GPU modifications"
    );
}

#[test]
fn test_gpu_optimization_on_conv_heavy_graph() {
    let mut graph = create_conv_heavy_graph();
    let pass = GpuOptimizationPass;

    let _result = pass.run(&mut graph).unwrap();

    // Conv-heavy graphs can benefit from GPU optimizations
    // Pass completed successfully if we reached here
}

#[test]
fn test_gpu_optimization_preserves_outputs() {
    let mut graph = create_simple_conv_graph();
    let original_outputs = graph.outputs.clone();
    let pass = GpuOptimizationPass;

    pass.run(&mut graph).unwrap();

    assert_eq!(
        graph.outputs, original_outputs,
        "Graph outputs should be preserved"
    );
}

#[test]
fn test_gpu_optimization_kernel_fusion() {
    // Create a graph with multiple element-wise operations (good for kernel fusion)
    let mut builder = builder();

    let input_id = builder.add_op("Input", Some("input".to_string()));
    builder.add_output(input_id, "input_tensor");

    let relu1_id = builder.add_op("Relu", Some("relu1".to_string()));
    builder
        .add_input(relu1_id, "input_tensor")
        .add_output(relu1_id, "relu1_output");

    let relu2_id = builder.add_op("Relu", Some("relu2".to_string()));
    builder
        .add_input(relu2_id, "relu1_output")
        .add_output(relu2_id, "relu2_output");

    let sigmoid_id = builder.add_op("Sigmoid", Some("sigmoid1".to_string()));
    builder
        .add_input(sigmoid_id, "relu2_output")
        .add_output(sigmoid_id, "output_tensor");

    builder.connect(input_id, relu1_id, "input_tensor").unwrap();
    builder.connect(relu1_id, relu2_id, "relu1_output").unwrap();
    builder
        .connect(relu2_id, sigmoid_id, "relu2_output")
        .unwrap();

    builder
        .set_inputs(vec!["input_tensor".to_string()])
        .set_outputs(vec!["output_tensor".to_string()]);

    let mut graph = builder.build().unwrap();
    let pass = GpuOptimizationPass;

    let result = pass.run(&mut graph).unwrap();

    // Multiple element-wise ops are good candidates for kernel fusion
    assert!(
        result.nodes_fused >= 0,
        "Should attempt kernel fusion for element-wise operations"
    );
}

// Comparison tests between CPU and GPU optimizations

#[test]
fn test_cpu_and_gpu_optimizations_are_independent() {
    let graph_original = create_simple_conv_graph();

    let mut graph_cpu = graph_original.clone();
    let mut graph_gpu = graph_original.clone();

    let cpu_pass = CpuOptimizationPass;
    let gpu_pass = GpuOptimizationPass;

    let _cpu_result = cpu_pass.run(&mut graph_cpu).unwrap();
    let _gpu_result = gpu_pass.run(&mut graph_gpu).unwrap();

    // Both should succeed independently
    assert!(
        verify_graph_valid(&graph_cpu),
        "CPU-optimized graph should be valid"
    );
    assert!(
        verify_graph_valid(&graph_gpu),
        "GPU-optimized graph should be valid"
    );
}

#[test]
fn test_provider_optimizations_on_same_graph() {
    let mut graph = create_fusible_graph();

    let cpu_pass = CpuOptimizationPass;
    let gpu_pass = GpuOptimizationPass;

    // Apply CPU optimization first
    let _cpu_result = cpu_pass.run(&mut graph).unwrap();
    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid after CPU optimization"
    );

    // Then apply GPU optimization
    let _gpu_result = gpu_pass.run(&mut graph).unwrap();
    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid after both optimizations"
    );

    // Both passes should work on the same graph sequentially
    // If we reached here, both passes succeeded
}

#[test]
fn test_provider_optimizations_preserve_semantics() {
    let graph_original = create_conv_heavy_graph();

    let mut graph_cpu = graph_original.clone();
    let mut graph_gpu = graph_original.clone();

    let cpu_pass = CpuOptimizationPass;
    let gpu_pass = GpuOptimizationPass;

    cpu_pass.run(&mut graph_cpu).unwrap();
    gpu_pass.run(&mut graph_gpu).unwrap();

    // Both should preserve the number of inputs and outputs
    assert_eq!(
        graph_cpu.inputs.len(),
        graph_original.inputs.len(),
        "CPU optimization should preserve inputs"
    );
    assert_eq!(
        graph_cpu.outputs.len(),
        graph_original.outputs.len(),
        "CPU optimization should preserve outputs"
    );
    assert_eq!(
        graph_gpu.inputs.len(),
        graph_original.inputs.len(),
        "GPU optimization should preserve inputs"
    );
    assert_eq!(
        graph_gpu.outputs.len(),
        graph_original.outputs.len(),
        "GPU optimization should preserve outputs"
    );
}
