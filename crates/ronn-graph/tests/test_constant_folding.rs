// Tests for constant folding optimization pass

mod common;

use common::*;
use ronn_graph::{ConstantFoldingPass, OptimizationPass};

#[test]
fn test_constant_folding_pass_runs() {
    let mut graph = create_constant_graph();
    let pass = ConstantFoldingPass;

    let result = pass.run(&mut graph);
    assert!(
        result.is_ok(),
        "Constant folding pass should complete successfully"
    );
}

#[test]
fn test_constant_folding_on_empty_graph() {
    let mut graph = create_empty_graph();
    let pass = ConstantFoldingPass;

    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should handle empty graph gracefully");

    let stats = result.unwrap();
    assert_eq!(
        stats.nodes_modified, 0,
        "Empty graph should have no modifications"
    );
}

#[test]
fn test_constant_folding_preserves_graph_validity() {
    let mut graph = create_constant_graph();
    let pass = ConstantFoldingPass;

    assert!(
        verify_graph_valid(&graph),
        "Graph should be valid before optimization"
    );

    pass.run(&mut graph).unwrap();

    assert!(
        verify_graph_valid(&graph),
        "Graph should remain valid after optimization"
    );
}

#[test]
fn test_constant_folding_on_non_constant_graph() {
    let mut graph = create_simple_conv_graph();
    let pass = ConstantFoldingPass;

    let result = pass.run(&mut graph).unwrap();

    // Should not modify non-constant operations
    assert_eq!(
        result.nodes_modified, 0,
        "Non-constant graph should not be modified"
    );
}

#[test]
fn test_constant_folding_identifies_foldable_ops() {
    let mut graph = create_constant_graph();
    let initial_node_count = graph.node_count();

    let pass = ConstantFoldingPass;
    let _result = pass.run(&mut graph).unwrap();

    // Graph structure should remain valid
    assert_eq!(
        graph.node_count(),
        initial_node_count,
        "Node count should not change (folding marks nodes, doesn't remove yet)"
    );
}

#[test]
fn test_constant_folding_pass_name() {
    let pass = ConstantFoldingPass;
    assert_eq!(pass.name(), "ConstantFolding");
}

#[test]
fn test_constant_folding_stats() {
    let mut graph = create_constant_graph();
    let pass = ConstantFoldingPass;

    let stats = pass.run(&mut graph).unwrap();

    // Verify stats structure is valid
    assert_eq!(
        stats.nodes_removed, 0,
        "Pass should not remove nodes directly"
    );
    assert_eq!(stats.nodes_fused, 0, "Pass should not fuse nodes");
    // nodes_modified may be > 0 if constants are found and marked
}

#[test]
fn test_constant_folding_idempotent() {
    let mut graph = create_constant_graph();
    let pass = ConstantFoldingPass;

    let result1 = pass.run(&mut graph).unwrap();
    let result2 = pass.run(&mut graph).unwrap();

    // Running twice should not cause additional changes
    assert_eq!(
        result1.nodes_modified, result2.nodes_modified,
        "Pass should be idempotent"
    );
}

#[test]
fn test_constant_folding_with_complex_graph() {
    let mut graph = create_fusible_graph();
    let pass = ConstantFoldingPass;

    let result = pass.run(&mut graph);
    assert!(result.is_ok(), "Should handle complex graphs");
}
