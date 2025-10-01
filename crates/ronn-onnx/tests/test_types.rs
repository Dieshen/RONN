//! Unit tests for ONNX type system
//! Tests: DataTypeMapper, type conversions, shape inference

use ronn_core::types::DataType;
use ronn_onnx::DataTypeMapper;

// ============ DataType Mapping Tests ============

#[test]
fn test_from_onnx_float_types() {
    // Test floating point types
    assert_eq!(
        DataTypeMapper::from_onnx(1).unwrap(),
        DataType::F32,
        "ONNX FLOAT (1) should map to F32"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(10).unwrap(),
        DataType::F16,
        "ONNX FLOAT16 (10) should map to F16"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(11).unwrap(),
        DataType::F64,
        "ONNX DOUBLE (11) should map to F64"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(16).unwrap(),
        DataType::BF16,
        "ONNX BFLOAT16 (16) should map to BF16"
    );
}

#[test]
fn test_from_onnx_integer_types() {
    // Test integer types
    assert_eq!(
        DataTypeMapper::from_onnx(2).unwrap(),
        DataType::U8,
        "ONNX UINT8 (2) should map to U8"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(3).unwrap(),
        DataType::I8,
        "ONNX INT8 (3) should map to I8"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(6).unwrap(),
        DataType::I32,
        "ONNX INT32 (6) should map to I32"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(7).unwrap(),
        DataType::I64,
        "ONNX INT64 (7) should map to I64"
    );

    assert_eq!(
        DataTypeMapper::from_onnx(12).unwrap(),
        DataType::U32,
        "ONNX UINT32 (12) should map to U32"
    );
}

#[test]
fn test_from_onnx_bool_type() {
    assert_eq!(
        DataTypeMapper::from_onnx(9).unwrap(),
        DataType::Bool,
        "ONNX BOOL (9) should map to Bool"
    );
}

#[test]
fn test_from_onnx_unsupported_types() {
    // Test unsupported types
    let unsupported = vec![
        0,  // UNDEFINED
        4,  // UINT16
        5,  // INT16
        8,  // STRING
        13, // UINT64
        14, // COMPLEX64
        15, // COMPLEX128
        99, // Invalid
    ];

    for onnx_type in unsupported {
        let result = DataTypeMapper::from_onnx(onnx_type);
        assert!(
            result.is_err(),
            "Type {} should be unsupported",
            onnx_type
        );
    }
}

#[test]
fn test_to_onnx_all_types() {
    // Test round-trip conversion for all supported types
    let types = vec![
        (DataType::F32, 1),
        (DataType::U8, 2),
        (DataType::I8, 3),
        (DataType::I32, 6),
        (DataType::I64, 7),
        (DataType::Bool, 9),
        (DataType::F16, 10),
        (DataType::F64, 11),
        (DataType::U32, 12),
        (DataType::BF16, 16),
    ];

    for (ronn_type, expected_onnx) in types {
        let onnx_type = DataTypeMapper::to_onnx(ronn_type);
        assert_eq!(
            onnx_type, expected_onnx,
            "{:?} should map to {}",
            ronn_type, expected_onnx
        );
    }
}

#[test]
fn test_round_trip_conversion() {
    // Test that from_onnx(to_onnx(x)) == x for all supported types
    let types = vec![
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::F64,
        DataType::U8,
        DataType::U32,
        DataType::I8,
        DataType::I32,
        DataType::I64,
        DataType::Bool,
    ];

    for ronn_type in types {
        let onnx_type = DataTypeMapper::to_onnx(ronn_type);
        let converted_back = DataTypeMapper::from_onnx(onnx_type).unwrap();
        assert_eq!(
            converted_back, ronn_type,
            "Round-trip conversion failed for {:?}",
            ronn_type
        );
    }
}

// ============ Proto Type Constants Tests ============
// Note: Proto module is private, so we skip direct constant testing
// These constants are verified indirectly through DataTypeMapper tests

// ============ Type Validation Tests ============

#[test]
fn test_all_ronn_types_mappable() {
    // Ensure all RONN DataType variants can be mapped to ONNX
    let all_types = vec![
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::F64,
        DataType::U8,
        DataType::U32,
        DataType::I8,
        DataType::I32,
        DataType::I64,
        DataType::Bool,
    ];

    for dtype in all_types {
        let onnx_type = DataTypeMapper::to_onnx(dtype);
        // Should be a valid ONNX type code
        assert!(onnx_type > 0 && onnx_type < 20, "Invalid ONNX type code for {:?}", dtype);
    }
}

// ============ Edge Cases and Error Handling ============

#[test]
fn test_negative_type_code() {
    let result = DataTypeMapper::from_onnx(-1);
    assert!(result.is_err(), "Negative type codes should fail");
}

#[test]
fn test_large_type_code() {
    let result = DataTypeMapper::from_onnx(1000);
    assert!(result.is_err(), "Large invalid type codes should fail");
}

#[test]
fn test_zero_type_code() {
    // UNDEFINED type
    let result = DataTypeMapper::from_onnx(0);
    assert!(result.is_err(), "UNDEFINED type (0) should not be supported");
}

// ============ Type Consistency Tests ============

#[test]
fn test_type_mapping_consistency() {
    // Ensure there are no duplicate mappings
    use std::collections::HashSet;

    let ronn_types = vec![
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::F64,
        DataType::U8,
        DataType::U32,
        DataType::I8,
        DataType::I32,
        DataType::I64,
        DataType::Bool,
    ];

    let mut onnx_codes = HashSet::new();

    for ronn_type in ronn_types {
        let onnx_code = DataTypeMapper::to_onnx(ronn_type);
        assert!(
            onnx_codes.insert(onnx_code),
            "Duplicate ONNX code {} for {:?}",
            onnx_code,
            ronn_type
        );
    }
}

// ============ Type Compatibility Tests ============

#[test]
fn test_floating_point_type_family() {
    let float_types = vec![
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::F64,
    ];

    for dtype in float_types {
        let onnx_code = DataTypeMapper::to_onnx(dtype);
        // Float types: 1, 10, 11, 16
        assert!(
            onnx_code == 1 || onnx_code == 10 || onnx_code == 11 || onnx_code == 16,
            "{:?} should map to a float ONNX code",
            dtype
        );
    }
}

#[test]
fn test_integer_type_family() {
    let int_types = vec![
        DataType::U8,
        DataType::U32,
        DataType::I8,
        DataType::I32,
        DataType::I64,
    ];

    for dtype in int_types {
        let onnx_code = DataTypeMapper::to_onnx(dtype);
        // Integer types: 2, 3, 6, 7, 12
        assert!(
            onnx_code == 2 || onnx_code == 3 || onnx_code == 6 || onnx_code == 7 || onnx_code == 12,
            "{:?} should map to an integer ONNX code",
            dtype
        );
    }
}

// ============ Documentation and Spec Compliance ============

#[test]
fn test_onnx_spec_version_compatibility() {
    // Test that our type mappings are compatible with ONNX IR version 7+
    // (which is the minimum version most modern models use)

    // Core types that must be supported
    let core_types = vec![
        1,  // FLOAT
        2,  // UINT8
        3,  // INT8
        6,  // INT32
        7,  // INT64
        9,  // BOOL
        10, // FLOAT16
        11, // DOUBLE
    ];

    for onnx_type in core_types {
        let result = DataTypeMapper::from_onnx(onnx_type);
        assert!(
            result.is_ok(),
            "Core ONNX type {} should be supported",
            onnx_type
        );
    }
}

// ============ Error Message Tests ============

#[test]
fn test_error_message_quality() {
    let result = DataTypeMapper::from_onnx(99);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = err.to_string();

    // Error message should contain the invalid type code
    assert!(
        err_msg.contains("99"),
        "Error message should include the invalid type code"
    );

    // Error message should mention type conversion
    assert!(
        err_msg.to_lowercase().contains("type") || err_msg.to_lowercase().contains("unsupported"),
        "Error message should mention type or unsupported"
    );
}

// ============ Tensor Proto Type Tests ============
// Verified indirectly through DataTypeMapper which uses proto types internally

// ============ Type Conversion Coverage ============

#[test]
fn test_all_supported_types_covered() {
    // Ensure we test conversion for all supported types
    let supported_onnx_types = vec![1, 2, 3, 6, 7, 9, 10, 11, 12, 16];

    for onnx_type in supported_onnx_types {
        let result = DataTypeMapper::from_onnx(onnx_type);
        assert!(
            result.is_ok(),
            "Type {} should be supported but conversion failed",
            onnx_type
        );

        // Verify round-trip
        let ronn_type = result.unwrap();
        let back_to_onnx = DataTypeMapper::to_onnx(ronn_type);
        assert_eq!(
            back_to_onnx, onnx_type,
            "Round-trip failed for ONNX type {}",
            onnx_type
        );
    }
}

#[test]
fn test_type_mapper_deterministic() {
    // Ensure type mapping is deterministic (always returns same result)
    let test_cases = vec![
        (1, DataType::F32),
        (10, DataType::F16),
        (16, DataType::BF16),
    ];

    for (onnx_type, expected_ronn) in test_cases {
        for _ in 0..10 {
            let result = DataTypeMapper::from_onnx(onnx_type).unwrap();
            assert_eq!(result, expected_ronn, "Type mapping should be deterministic");
        }
    }
}
