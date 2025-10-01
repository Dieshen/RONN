//! Unit tests for ONNX model loader
//! Tests: Model parsing, graph conversion, attribute extraction, initializer loading

use ronn_core::NodeAttribute;
use ronn_onnx::ModelLoader;
use serde_json::json;

// ============ Model Loading Tests ============

#[test]
fn test_load_minimal_model() {
    // Create minimal valid ONNX model
    let model_json = json!({
        "ir_version": 7,
        "producer_name": "test",
        "graph": {
            "name": "test_graph",
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should load minimal model");

    let model = result.unwrap();
    assert_eq!(model.ir_version, 7);
    assert_eq!(model.producer_name, Some("test".to_string()));
}

#[test]
fn test_load_model_with_nodes() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test_graph",
            "node": [
                {
                    "name": "relu_node",
                    "op_type": "Relu",
                    "input": ["input1"],
                    "output": ["output1"],
                    "attribute": []
                }
            ],
            "input": [{
                "name": "input1",
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
                "name": "output1",
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
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.inputs[0].name, "input1");
    assert_eq!(model.inputs[0].shape, vec![1, 3, 224, 224]);

    assert_eq!(model.outputs.len(), 1);
    assert_eq!(model.outputs[0].name, "output1");
}

#[test]
fn test_load_model_with_initializers() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test_graph",
            "node": [],
            "input": [
                {
                    "name": "input1",
                    "type": {
                        "value": {
                            "tensor_type": {
                                "elem_type": 1,
                                "shape": {"dim": [{"value": {"dim_value": 2}}]}
                            }
                        }
                    }
                },
                {
                    "name": "weight",
                    "type": {
                        "value": {
                            "tensor_type": {
                                "elem_type": 1,
                                "shape": {"dim": [{"value": {"dim_value": 2}}]}
                            }
                        }
                    }
                }
            ],
            "output": [],
            "initializer": [
                {
                    "name": "weight",
                    "dims": [2],
                    "data_type": 1,
                    "float_data": [1.0, 2.0],
                    "int32_data": [],
                    "int64_data": [],
                    "raw_data": [],
                    "double_data": [],
                    "uint64_data": []
                }
            ]
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    // Weight should be in initializers, not in inputs
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.inputs[0].name, "input1");

    assert!(model.initializers.contains_key("weight"));
    assert_eq!(model.initializers.len(), 1);
}

#[test]
fn test_load_invalid_json() {
    let invalid_json = b"{ invalid json }";
    let result = ModelLoader::load_from_bytes(invalid_json);

    assert!(result.is_err());
}

#[test]
fn test_load_model_without_graph() {
    let model_json = json!({
        "ir_version": 7,
        "producer_name": "test"
        // Missing graph field
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_err(), "Should fail without graph");
}

// ============ Attribute Parsing Tests ============

#[test]
fn test_parse_float_attribute() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "test_node",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [{
                    "name": "alpha",
                    "f": 0.5,
                    "type": 1,  // FLOAT
                    "floats": [],
                    "ints": [],
                    "strings": []
                }]
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    assert_eq!(nodes.len(), 1);

    let node = &nodes[0];
    assert!(node.attributes.contains_key("alpha"));
    if let Some(NodeAttribute::Float(val)) = node.attributes.get("alpha") {
        assert!((val - 0.5).abs() < 1e-6);
    } else {
        panic!("Expected Float attribute");
    }
}

#[test]
fn test_parse_int_attribute() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "test_node",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [{
                    "name": "axis",
                    "i": 1,
                    "type": 2,  // INT
                    "floats": [],
                    "ints": [],
                    "strings": []
                }]
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::Int(val)) = node.attributes.get("axis") {
        assert_eq!(*val, 1);
    } else {
        panic!("Expected Int attribute");
    }
}

#[test]
fn test_parse_string_attribute() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "test_node",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [{
                    "name": "mode",
                    "s": [116, 101, 115, 116],  // "test" in bytes
                    "type": 3,  // STRING
                    "floats": [],
                    "ints": [],
                    "strings": []
                }]
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::String(val)) = node.attributes.get("mode") {
        assert_eq!(val, "test");
    } else {
        panic!("Expected String attribute");
    }
}

#[test]
fn test_parse_floats_attribute() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "test_node",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [{
                    "name": "scales",
                    "floats": [1.0, 2.0, 3.0],
                    "type": 6,  // FLOATS
                    "ints": [],
                    "strings": []
                }]
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::FloatArray(vals)) = node.attributes.get("scales") {
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
        assert!((vals[2] - 3.0).abs() < 1e-6);
    } else {
        panic!("Expected FloatArray attribute");
    }
}

#[test]
fn test_parse_ints_attribute() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "test_node",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [{
                    "name": "pads",
                    "ints": [1, 1, 1, 1],
                    "type": 7,  // INTS
                    "floats": [],
                    "strings": []
                }]
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::IntArray(vals)) = node.attributes.get("pads") {
        assert_eq!(vals, &[1, 1, 1, 1]);
    } else {
        panic!("Expected IntArray attribute");
    }
}

#[test]
fn test_parse_multiple_attributes() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "conv_node",
                "op_type": "Conv",
                "input": ["input", "weight"],
                "output": ["output"],
                "attribute": [
                    {
                        "name": "group",
                        "i": 1,
                        "type": 2,
                        "floats": [],
                        "ints": [],
                        "strings": []
                    },
                    {
                        "name": "strides",
                        "ints": [1, 1],
                        "type": 7,
                        "floats": [],
                        "strings": []
                    },
                    {
                        "name": "pads",
                        "ints": [0, 0, 0, 0],
                        "type": 7,
                        "floats": [],
                        "strings": []
                    }
                ]
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    assert_eq!(node.attributes.len(), 3);
    assert!(node.attributes.contains_key("group"));
    assert!(node.attributes.contains_key("strides"));
    assert!(node.attributes.contains_key("pads"));
}

// ============ Shape Inference Tests ============

#[test]
fn test_shape_inference_static() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
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
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.inputs[0].shape, vec![1, 3, 224, 224]);
}

#[test]
fn test_shape_inference_dynamic() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
            "input": [{
                "name": "input",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_param": "batch"}},
                                    {"value": {"dim_value": 3}},
                                    {"value": {"dim_value": 224}},
                                    {"value": {"dim_value": 224}}
                                ]
                            }
                        }
                    }
                }
            }],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    // Dynamic dimensions are represented as 0
    assert_eq!(model.inputs[0].shape, vec![0, 3, 224, 224]);
}

// ============ Graph Structure Tests ============

#[test]
fn test_node_name_generation() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [
                {
                    // Node without name - should be auto-generated
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
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();

    // Should have generated a name like "node_0"
    assert!(nodes[0].name.is_some());
}

#[test]
fn test_multiple_nodes() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [
                {
                    "name": "relu1",
                    "op_type": "Relu",
                    "input": ["input"],
                    "output": ["relu_out"],
                    "attribute": []
                },
                {
                    "name": "add1",
                    "op_type": "Add",
                    "input": ["relu_out", "bias"],
                    "output": ["output"],
                    "attribute": []
                }
            ],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();

    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].op_type, "Relu");
    assert_eq!(nodes[1].op_type, "Add");
}

// ============ Version Compatibility Tests ============

#[test]
fn test_various_ir_versions() {
    let versions = vec![3, 4, 5, 6, 7, 8, 9];

    for version in versions {
        let model_json = json!({
            "ir_version": version,
            "graph": {
                "name": "test",
                "node": [],
                "input": [],
                "output": [],
                "initializer": []
            }
        });

        let model_bytes = serde_json::to_vec(&model_json).unwrap();
        let result = ModelLoader::load_from_bytes(&model_bytes);

        assert!(result.is_ok(), "Should support IR version {}", version);

        let model = result.unwrap();
        assert_eq!(model.ir_version, version);
    }
}

// ============ Edge Cases ============

#[test]
fn test_empty_graph() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "empty",
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.inputs.len(), 0);
    assert_eq!(model.outputs.len(), 0);
    let nodes = model.graph.nodes();
    assert_eq!(nodes.len(), 0);
}

#[test]
fn test_producer_name_optional() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
        // No producer_name field
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.producer_name, None);
}
