#!/usr/bin/env python3
"""
Generate a tiny ONNX model for testing RONN's ONNX loader.
This creates a simple Add operation model.

Requirements: pip install onnx numpy
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_simple_add_model():
    """Create a simple model that adds two inputs."""

    # Define inputs
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

    # Define output
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 4])

    # Create the Add node
    add_node = helper.make_node(
        'Add',                # Op type
        inputs=['X', 'Y'],    # Input names
        outputs=['Z'],        # Output names
        name='add_node'       # Node name
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[add_node],
        name='simple_add_graph',
        inputs=[X, Y],
        outputs=[Z],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name='ronn',
        opset_imports=[helper.make_opsetid("", 13)]
    )

    # Check the model
    onnx.checker.check_model(model_def)

    return model_def

def create_simple_matmul_model():
    """Create a simple matrix multiplication model."""

    # Define inputs (2x3 and 3x2 matrices)
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 2])

    # Define output (2x2 matrix)
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2, 2])

    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['C'],
        name='matmul_node'
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[matmul_node],
        name='simple_matmul_graph',
        inputs=[A, B],
        outputs=[C],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name='ronn',
        opset_imports=[helper.make_opsetid("", 13)]
    )

    # Check the model
    onnx.checker.check_model(model_def)

    return model_def

def create_relu_model():
    """Create a simple ReLU activation model."""

    # Define input
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])

    # Define output
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

    # Create ReLU node
    relu_node = helper.make_node(
        'Relu',
        inputs=['X'],
        outputs=['Y'],
        name='relu_node'
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[relu_node],
        name='simple_relu_graph',
        inputs=[X],
        outputs=[Y],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name='ronn',
        opset_imports=[helper.make_opsetid("", 13)]
    )

    # Check the model
    onnx.checker.check_model(model_def)

    return model_def

def main():
    import os

    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Generate models
    models = {
        'simple_add.onnx': create_simple_add_model(),
        'simple_matmul.onnx': create_simple_matmul_model(),
        'simple_relu.onnx': create_relu_model(),
    }

    for filename, model in models.items():
        filepath = os.path.join(models_dir, filename)
        onnx.save(model, filepath)
        print(f"✅ Generated: {filepath}")
        print(f"   - Ops: {[node.op_type for node in model.graph.node]}")
        print(f"   - Inputs: {[inp.name for inp in model.graph.input]}")
        print(f"   - Outputs: {[out.name for out in model.graph.output]}")
        print()

if __name__ == '__main__':
    main()
