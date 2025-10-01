#!/usr/bin/env python3
"""
Generate simple ONNX test fixtures for ronn-api tests.

This script creates minimal ONNX models suitable for testing the RONN API.
"""

import sys

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install onnx numpy")
    sys.exit(1)


def create_simple_identity_model():
    """Create a simple identity model: output = input"""

    # Input tensor
    input_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 10])

    # Output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

    # Identity node
    node = helper.make_node(
        'Identity',
        inputs=['data'],
        outputs=['output'],
        name='identity_node'
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[node],
        name='SimpleIdentityModel',
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    # Create model
    model = helper.make_model(graph, producer_name='ronn-test-fixture')
    model.opset_import[0].version = 13
    model.ir_version = 8

    # Verify model
    try:
        onnx.checker.check_model(model)
        print("[OK] Model validation passed")
    except Exception as e:
        print(f"[ERROR] Model validation failed: {e}")
        return None

    return model


def create_simple_add_model():
    """Create a model that adds a constant: output = input + 1.0"""

    # Input tensor
    input_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 10])

    # Output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

    # Constant tensor (value to add)
    constant_value = np.array([1.0], dtype=np.float32)
    constant_tensor = numpy_helper.from_array(constant_value, name='constant')

    # Add node
    node = helper.make_node(
        'Add',
        inputs=['data', 'constant'],
        outputs=['output'],
        name='add_node'
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[node],
        name='SimpleAddModel',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[constant_tensor]
    )

    # Create model
    model = helper.make_model(graph, producer_name='ronn-test-fixture')
    model.opset_import[0].version = 13
    model.ir_version = 8

    # Verify model
    try:
        onnx.checker.check_model(model)
        print("[OK] Model validation passed")
    except Exception as e:
        print(f"[ERROR] Model validation failed: {e}")
        return None

    return model


def main():
    print("Creating ONNX test fixtures for ronn-api...\n")

    # Create simple identity model (primary test fixture)
    print("1. Creating simple_model.onnx (Identity operation)...")
    model = create_simple_identity_model()
    if model:
        onnx.save(model, 'simple_model.onnx')
        import os
        size = os.path.getsize('simple_model.onnx')
        print(f"   [OK] Saved simple_model.onnx ({size} bytes)\n")
    else:
        print("   [ERROR] Failed to create simple_model.onnx\n")
        return 1

    # Create add model (alternative test fixture)
    print("2. Creating add_model.onnx (Add operation)...")
    model = create_simple_add_model()
    if model:
        onnx.save(model, 'add_model.onnx')
        import os
        size = os.path.getsize('add_model.onnx')
        print(f"   [OK] Saved add_model.onnx ({size} bytes)\n")
    else:
        print("   [ERROR] Failed to create add_model.onnx\n")

    print("[SUCCESS] Test fixtures created successfully!")
    print("\nTo run tests with fixtures:")
    print("  cd crates/ronn-api")
    print("  cargo test -- --include-ignored")

    return 0


if __name__ == '__main__':
    sys.exit(main())
