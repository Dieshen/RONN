//! Property-based and fuzzing tests for ONNX parser robustness
//! Tests: Malformed inputs, boundary conditions, invalid protobuf data

use ronn_onnx::ModelLoader;
use serde_json::json;

// ============ Malformed Input Tests ============

#[test]
fn test_empty_input() {
    let result = ModelLoader::load_from_bytes(&[]);
    assert!(result.is_err(), "Should reject empty input");
}

#[test]
fn test_random_bytes() {
    let random_data = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA];
    let result = ModelLoader::load_from_bytes(&random_data);
    assert!(result.is_err(), "Should reject random bytes");
}

#[test]
fn test_partial_json() {
    let partial_json = b"{\"ir_version\": 7, \"graph\":";
    let result = ModelLoader::load_from_bytes(partial_json);
    assert!(result.is_err(), "Should reject incomplete JSON");
}

#[test]
fn test_invalid_unicode() {
    let invalid_utf8 = vec![
        b'{', b'"', b'n', b'a', b'm', b'e', b'"', b':', b'"',
        0xFF, 0xFE, // Invalid UTF-8 sequence
        b'"', b'}',
    ];
    let result = ModelLoader::load_from_bytes(&invalid_utf8);
    assert!(result.is_err(), "Should reject invalid UTF-8");
}

// ============ Missing Required Fields ============

#[test]
fn test_missing_graph_name() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            // Missing "name" field
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should still load - name is optional
    assert!(result.is_ok());
}

#[test]
fn test_missing_node_op_type() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "node1",
                // Missing "op_type"
                "input": [],
                "output": [],
                "attribute": []
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should still load with empty op_type
    assert!(result.is_ok());
}

#[test]
fn test_missing_tensor_dims() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
            "input": [],
            "output": [],
            "initializer": [{
                "name": "weight",
                // Missing "dims"
                "data_type": 1,
                "float_data": [1.0, 2.0],
                "int32_data": [],
                "int64_data": [],
                "raw_data": [],
                "double_data": [],
                "uint64_data": []
            }]
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // May fail during tensor creation
    // Just ensure it doesn't panic
    let _ = result;
}

// ============ Invalid Type Combinations ============

#[test]
fn test_attribute_type_mismatch() {
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
                    "name": "value",
                    "type": 1,  // FLOAT
                    "i": 42,    // But providing int value
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

    // Should handle gracefully (use 0.0 as default)
    assert!(result.is_ok());
}

#[test]
fn test_invalid_data_type() {
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
                            "elem_type": 999,  // Invalid type
                            "shape": {"dim": [{"value": {"dim_value": 2}}]}
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

    // Should load but may have issues with type mapping
    // Just ensure no panic
    let _ = result;
}

// ============ Boundary Conditions ============

#[test]
fn test_very_large_tensor_dims() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
            "input": [{
                "name": "huge_input",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": 1000000}},
                                    {"value": {"dim_value": 1000000}}
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

    // Should be able to parse the shape without allocating tensor
    assert!(result.is_ok());
}

#[test]
fn test_zero_dimensions() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
            "input": [{
                "name": "zero_dim",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": 0}}
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
    assert_eq!(model.inputs[0].shape, vec![0]);
}

#[test]
fn test_negative_dimensions() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [],
            "input": [{
                "name": "negative_dim",
                "type": {
                    "value": {
                        "tensor_type": {
                            "elem_type": 1,
                            "shape": {
                                "dim": [
                                    {"value": {"dim_value": -1}}
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

    // Negative dimensions might be interpreted as large positive numbers
    // Should not panic
    let _ = result;
}

// ============ Deep Nesting ============

#[test]
fn test_deeply_nested_graph() {
    // Create a model with many nodes
    let mut nodes = Vec::new();
    for i in 0..100 {
        nodes.push(json!({
            "name": format!("node_{}", i),
            "op_type": "Relu",
            "input": [format!("input_{}", i)],
            "output": [format!("output_{}", i)],
            "attribute": []
        }));
    }

    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "deep_graph",
            "node": nodes,
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle many nodes");

    let model = result.unwrap();
    let node_count = model.graph.nodes();
    assert_eq!(node_count.len(), 100);
}

// ============ String Handling ============

#[test]
fn test_very_long_names() {
    let long_name = "a".repeat(10000);

    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": long_name,
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle long names");
}

#[test]
fn test_empty_string_names() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "",
            "node": [{
                "name": "",
                "op_type": "",
                "input": [""],
                "output": [""],
                "attribute": []
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle empty strings");
}

#[test]
fn test_special_characters_in_names() {
    let special_chars = "!@#$%^&*()[]{}|\\:;'\"<>,.?/~`";

    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": special_chars,
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle special characters");
}

// ============ Array Handling ============

#[test]
fn test_empty_arrays() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "test_node",
                "op_type": "Test",
                "input": [],  // Empty input array
                "output": [], // Empty output array
                "attribute": [{
                    "name": "empty_ints",
                    "ints": [],
                    "type": 7,
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
}

#[test]
fn test_very_large_arrays() {
    let large_array: Vec<i64> = (0..10000).collect();

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
                    "name": "large_ints",
                    "ints": large_array,
                    "type": 7,
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

    assert!(result.is_ok(), "Should handle large arrays");
}

// ============ Circular Reference Prevention ============

#[test]
fn test_node_output_as_input() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "cycle",
                "op_type": "Add",
                "input": ["x", "y"],
                "output": ["x"],  // Output same as input
                "attribute": []
            }],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let model_bytes = serde_json::to_vec(&model_json).unwrap();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should parse but graph validation might catch issues
    assert!(result.is_ok());
}

// ============ Numeric Edge Cases ============

#[test]
fn test_extreme_float_values() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "extreme",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [
                    {
                        "name": "infinity",
                        "f": f32::INFINITY,
                        "type": 1,
                        "floats": [],
                        "ints": [],
                        "strings": []
                    },
                    {
                        "name": "neg_infinity",
                        "f": f32::NEG_INFINITY,
                        "type": 1,
                        "floats": [],
                        "ints": [],
                        "strings": []
                    },
                    {
                        "name": "nan",
                        "f": f32::NAN,
                        "type": 1,
                        "floats": [],
                        "ints": [],
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

    // JSON may not support infinity/nan, but should not panic
    let _ = result;
}

#[test]
fn test_extreme_int_values() {
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "test",
            "node": [{
                "name": "extreme",
                "op_type": "Test",
                "input": [],
                "output": [],
                "attribute": [
                    {
                        "name": "max_i64",
                        "i": i64::MAX,
                        "type": 2,
                        "floats": [],
                        "ints": [],
                        "strings": []
                    },
                    {
                        "name": "min_i64",
                        "i": i64::MIN,
                        "type": 2,
                        "floats": [],
                        "ints": [],
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

    assert!(result.is_ok(), "Should handle extreme integers");
}

// ============ Memory Safety ============

#[test]
fn test_large_file_size() {
    // Create a model with reasonable content but test parsing doesn't allocate excessively
    let model_json = json!({
        "ir_version": 7,
        "graph": {
            "name": "memory_test",
            "node": [],
            "input": [],
            "output": [],
            "initializer": []
        }
    });

    let mut model_bytes = serde_json::to_vec(&model_json).unwrap();

    // Pad with whitespace (won't allocate much in parser)
    model_bytes.extend(vec![b' '; 1000]);

    let result = ModelLoader::load_from_bytes(&model_bytes);
    assert!(result.is_ok());
}
