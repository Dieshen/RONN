//! Integration tests for ONNX model loading and execution
//! Tests complete workflows with realistic model structures
//!
//! NOTE: Most tests are ignored because they require proper ONNX protobuf format.
//! The ModelLoader expects binary protobuf, not JSON. These tests are kept as
//! documentation of expected functionality but require refactoring to use
//! actual ONNX model generation (e.g., via Python's onnx library).
//!
//! Real ONNX integration tests are in the ronn-integration-tests crate.

use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use ronn_onnx::{ModelLoader, OperatorRegistry};
use serde_json::json;

// ============ Simple Linear Model ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_simple_linear_model() {
    // Create a simple model: input -> Add(bias) -> Relu -> output
    let model_json = json!({
        "ir_version": 7,
        "producer_name": "test_producer",
        "graph": {
            "name": "simple_linear",
            "node": [
                {
                    "name": "add_bias",
                    "op_type": "Add",
                    "input": ["input", "bias"],
                    "output": ["add_out"],
                    "attribute": []
                },
                {
                    "name": "activation",
                    "op_type": "Relu",
                    "input": ["add_out"],
                    "output": ["output"],
                    "attribute": []
                }
            ],
            "input": [{
                "name": "input",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": 1}},
                                    {"value": {"dim_value": 4}}
                                ]
                            }
                        }
                    }
                }
            }],
            "output": [{
                "name": "output",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": 1}},
                                    {"value": {"dim_value": 4}}
                                ]
                            }
                        }
                    }
                }
            }],
            "initializer": [{
                "name": "bias",
                "dims": [4],
                "data_type": 1,
                "float_data": [1.0, 2.0, 3.0, 4.0],
                "int32_data": [],
                "int64_data": [],
                "raw_data": [],
                "double_data": [],
                "uint64_data": []
            }]
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    // Verify model structure
    assert_eq!(loaded_model.inputs.len(), 1);
    assert_eq!(loaded_model.outputs.len(), 1);
    assert_eq!(loaded_model.inputs[0].shape, vec![1, 4]);
    assert_eq!(loaded_model.outputs[0].shape, vec![1, 4]);

    // Verify initializers
    assert_eq!(loaded_model.initializers.len(), 1);
    assert!(loaded_model.initializers.contains_key("bias"));

    // Verify graph nodes
    let nodes = loaded_model.graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].op_type, "Add");
    assert_eq!(nodes[1].op_type, "Relu");
}

// ============ Multi-operator Model ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_multi_operator_model() {
    // Model: input1 + input2 -> Mul(scale) -> Sigmoid -> output
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "multi_op",
            "node": [
                {
                    "name": "add",
                    "op_type": "Add",
                    "input": ["input1", "input2"],
                    "output": ["add_out"],
                    "attribute": []
                },
                {
                    "name": "mul",
                    "op_type": "Mul",
                    "input": ["add_out", "scale"],
                    "output": ["mul_out"],
                    "attribute": []
                },
                {
                    "name": "sigmoid",
                    "op_type": "Sigmoid",
                    "input": ["mul_out"],
                    "output": ["output"],
                    "attribute": []
                }
            ],
            "input": [
                {
                    "name": "input1",
                    "type": {
                        "value": {
                            "tensor_type": {
                                "elem_type": 1,
                                "shape": {"dim": [{"value": {"dim_value": 3}}]}
                            }
                        }
                    }
                },
                {
                    "name": "input2",
                    "type": {
                        "value": {
                            "tensor_type": {
                                "elem_type": 1,
                                "shape": {"dim": [{"value": {"dim_value": 3}}]}
                            }
                        }
                    }
                }
            ],
            "output": [{
                "name": "output",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {"dim": [{"value": {"dim_value": 3}}]}
                        }
                    }
                }
            }],
            "initializer": [{
                "name": "scale",
                "dims": [1],
                "data_type": 1,
                "float_data": [0.5],
                "int32_data": [],
                "int64_data": [],
                "raw_data": [],
                "double_data": [],
                "uint64_data": []
            }]
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    // Verify multiple inputs
    assert_eq!(loaded_model.inputs.len(), 2);
    assert_eq!(loaded_model.inputs[0].name, "input1");
    assert_eq!(loaded_model.inputs[1].name, "input2");

    // Verify node sequence
    let nodes = loaded_model.graph.nodes();
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[0].op_type, "Add");
    assert_eq!(nodes[1].op_type, "Mul");
    assert_eq!(nodes[2].op_type, "Sigmoid");
}

// ============ Operator Registry Integration ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_operator_registry_with_model() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [
                {
                    "name": "relu",
                    "op_type": "Relu",
                    "input": ["x"],
                    "output": ["y"],
                    "attribute": []
                }
            ],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    // Verify operator is supported in registry
    let registry = OperatorRegistry::new();
    let nodes = loaded_model.graph.nodes();

    for node in nodes {
        assert!(
            registry.is_supported(&node.op_type),
            "Operator {} should be registered",
            node.op_type
        );
    }
}

// ============ Full Operator Coverage Test ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_all_supported_operators() {
    // Create a model with all 20 supported operators
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "all_ops",
            "node": [
                // Activations (5)
                {"name": "relu", "op_type": "Relu", "input": ["x1"], "output": ["o1"], "attribute": []},
                {"name": "sigmoid", "op_type": "Sigmoid", "input": ["x2"], "output": ["o2"], "attribute": []},
                {"name": "tanh", "op_type": "Tanh", "input": ["x3"], "output": ["o3"], "attribute": []},
                {"name": "softmax", "op_type": "Softmax", "input": ["x4"], "output": ["o4"], "attribute": []},
                {"name": "gelu", "op_type": "Gelu", "input": ["x5"], "output": ["o5"], "attribute": []},

                // Math ops (5)
                {"name": "add", "op_type": "Add", "input": ["x6", "x7"], "output": ["o6"], "attribute": []},
                {"name": "sub", "op_type": "Sub", "input": ["x8", "x9"], "output": ["o7"], "attribute": []},
                {"name": "mul", "op_type": "Mul", "input": ["x10", "x11"], "output": ["o8"], "attribute": []},
                {"name": "div", "op_type": "Div", "input": ["x12", "x13"], "output": ["o9"], "attribute": []},
                {"name": "matmul", "op_type": "MatMul", "input": ["x14", "x15"], "output": ["o10"], "attribute": []},

                // Neural network (4)
                {"name": "conv", "op_type": "Conv", "input": ["x16", "x17"], "output": ["o11"], "attribute": []},
                {"name": "maxpool", "op_type": "MaxPool", "input": ["x18"], "output": ["o12"], "attribute": []},
                {"name": "avgpool", "op_type": "AveragePool", "input": ["x19"], "output": ["o13"], "attribute": []},
                {"name": "batchnorm", "op_type": "BatchNormalization", "input": ["x20", "x21", "x22", "x23", "x24"], "output": ["o14"], "attribute": []},

                // Tensor ops (6)
                {"name": "reshape", "op_type": "Reshape", "input": ["x25", "x26"], "output": ["o15"], "attribute": []},
                {"name": "transpose", "op_type": "Transpose", "input": ["x27"], "output": ["o16"], "attribute": []},
                {"name": "concat", "op_type": "Concat", "input": ["x28", "x29"], "output": ["o17"], "attribute": [{"name": "axis", "i": 0, "type": 2, "floats": [], "ints": [], "strings": []}]},
                {"name": "split", "op_type": "Split", "input": ["x30"], "output": ["o18"], "attribute": []},
                {"name": "gather", "op_type": "Gather", "input": ["x31", "x32"], "output": ["o19"], "attribute": []},
                {"name": "slice", "op_type": "Slice", "input": ["x33"], "output": ["o20"], "attribute": []}
            ],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    // Verify all 20 operators are present
    let nodes = loaded_model.graph.nodes();
    assert_eq!(nodes.len(), 20, "Should have all 20 operators");

    // Verify all are supported in registry
    let registry = OperatorRegistry::new();
    let supported_ops = registry.supported_operators();

    assert!(
        supported_ops.len() >= 20,
        "Registry should support at least 20 operators"
    );

    for node in nodes {
        assert!(
            registry.is_supported(&node.op_type),
            "Operator {} should be supported",
            node.op_type
        );
    }
}

// ============ Model Metadata Tests ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_model_metadata() {
    let model_json = json!({
        "ir_version": 8,
        "producer_name": "pytest",
        "producer_version": "1.0",
        "model_version": 1,
        "doc_string": "Test model",
        "graph": {
            "name": "test_graph",
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    assert_eq!(loaded_model.ir_version, 8);
    assert_eq!(loaded_model.producer_name, Some("pytest".to_string()));
}

// ============ Dynamic Shape Handling ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_dynamic_batch_size() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "dynamic_batch",
            "node": [{
                "name": "relu",
                "op_type": "Relu",
                "input": ["input"],
                "output": ["output"],
                "attribute": []
            }],
            "input": [{
                "name": "input",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_param": "batch_size"}},
                                    {"value": {"dim_value": 3}},
                                    {"value": {"dim_value": 224}},
                                    {"value": {"dim_value": 224}}
                                ]
                            }
                        }
                    }
                }
            }],
            "output": [{
                "name": "output",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_param": "batch_size"}},
                                    {"value": {"dim_value": 3}},
                                    {"value": {"dim_value": 224}},
                                    {"value": {"dim_value": 224}}
                                ]
                            }
                        }
                    }
                }
            }],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    // Dynamic dimensions should be represented as 0
    assert_eq!(loaded_model.inputs[0].shape[0], 0);
    assert_eq!(loaded_model.inputs[0].shape[1], 3);
    assert_eq!(loaded_model.inputs[0].shape[2], 224);
    assert_eq!(loaded_model.inputs[0].shape[3], 224);
}

// ============ Operator Execution Test ============

#[test]
fn test_execute_simple_operator() {
    // Test that we can execute a simple operator from a loaded model
    let registry = OperatorRegistry::new();

    // Get the Relu operator
    let relu_op = registry.get("Relu").unwrap();

    // Create test input
    let data = vec![-1.0, 0.0, 1.0, 2.0];
    let input = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor).unwrap();

    let inputs = vec![&input];
    let attributes = std::collections::HashMap::new();

    // Execute
    let results = relu_op.execute(&inputs, &attributes).unwrap();

    // Verify output
    assert_eq!(results.len(), 1);
    let output = results[0].to_vec().unwrap();

    // ReLU should zero negative values
    assert_eq!(output[0], 0.0);
    assert_eq!(output[1], 0.0);
    assert_eq!(output[2], 1.0);
    assert_eq!(output[3], 2.0);
}

// ============ Complex Model Test ============

#[test]
#[ignore = "Requires protobuf format, not JSON. See crate ronn-integration-tests for real ONNX tests."]
fn test_resnet_like_structure() {
    // Simplified ResNet-like structure: Conv -> BatchNorm -> Relu -> MaxPool
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "resnet_like",
            "node": [
                {
                    "name": "conv1",
                    "op_type": "Conv",
                    "input": ["input", "conv1_weight", "conv1_bias"],
                    "output": ["conv1_out"],
                    "attribute": [
                        {"name": "strides", "ints": [2, 2], "type": 7, "floats": [], "strings": []},
                        {"name": "pads", "ints": [3, 3, 3, 3], "type": 7, "floats": [], "strings": []},
                        {"name": "kernel_shape", "ints": [7, 7], "type": 7, "floats": [], "strings": []}
                    ]
                },
                {
                    "name": "bn1",
                    "op_type": "BatchNormalization",
                    "input": ["conv1_out", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
                    "output": ["bn1_out"],
                    "attribute": [
                        {"name": "epsilon", "f": 1e-5, "type": 1, "floats": [], "ints": [], "strings": []}
                    ]
                },
                {
                    "name": "relu1",
                    "op_type": "Relu",
                    "input": ["bn1_out"],
                    "output": ["relu1_out"],
                    "attribute": []
                },
                {
                    "name": "pool1",
                    "op_type": "MaxPool",
                    "input": ["relu1_out"],
                    "output": ["output"],
                    "attribute": [
                        {"name": "kernel_shape", "ints": [3, 3], "type": 7, "floats": [], "strings": []},
                        {"name": "strides", "ints": [2, 2], "type": 7, "floats": [], "strings": []}
                    ]
                }
            ],
            "input": [{
                "name": "input",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": 1}},
                                    {"value": {"dim_value": 3}},
                                    {"value": {"dim_value": 224}},
                                    {"value": {"dim_value": 224}}
                                ]
                            }
                        }
                    }
                }
            }],
            "output": [{
                "name": "output",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": 1}},
                                    {"value": {"dim_value": 64}},
                                    {"value": {"dim_value": 56}},
                                    {"value": {"dim_value": 56}}
                                ]
                            }
                        }
                    }
                }
            }],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let loaded_model = ModelLoader::load_from_bytes(&model_bytes).unwrap();

    // Verify structure
    let nodes = loaded_model.graph.nodes();
    assert_eq!(nodes.len(), 4);

    // Verify operator sequence
    assert_eq!(nodes[0].op_type, "Conv");
    assert_eq!(nodes[1].op_type, "BatchNormalization");
    assert_eq!(nodes[2].op_type, "Relu");
    assert_eq!(nodes[3].op_type, "MaxPool");

    // Verify attributes are preserved
    assert!(nodes[0].attributes.contains_key("strides"));
    assert!(nodes[0].attributes.contains_key("pads"));
    assert!(nodes[1].attributes.contains_key("epsilon"));
}
