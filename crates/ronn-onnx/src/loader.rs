use crate::error::{OnnxError, Result};
use crate::proto;
use crate::types::{shape_from_value_info, tensor_from_proto, DataTypeMapper};
use ronn_core::{GraphNode, ModelGraph, NodeAttribute};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, info, warn};

/// Loads ONNX models and converts them to RONN internal representation
pub struct ModelLoader;

impl ModelLoader {
    /// Load an ONNX model from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<LoadedModel> {
        info!("Loading ONNX model from: {:?}", path.as_ref());
        let bytes = fs::read(path)?;
        Self::load_from_bytes(&bytes)
    }

    /// Load an ONNX model from bytes
    pub fn load_from_bytes(bytes: &[u8]) -> Result<LoadedModel> {
        // For MVP: Try JSON format first (simplified testing)
        // In production: Use prost to decode protobuf
        let model_proto: proto::ModelProto = serde_json::from_slice(bytes)
            .map_err(|e| OnnxError::ParseError(format!("Failed to parse model: {}", e)))?;
        Self::convert_model(model_proto)
    }

    /// Convert ONNX ModelProto to RONN LoadedModel
    fn convert_model(model_proto: proto::ModelProto) -> Result<LoadedModel> {
        let graph_proto = model_proto
            .graph
            .ok_or_else(|| OnnxError::ParseError("Model has no graph".to_string()))?;

        info!("Converting ONNX graph: {}", graph_proto.name.as_ref().unwrap_or(&"unnamed".to_string()));

        // Parse initializers (weights/constants)
        let mut initializers = HashMap::new();
        for init in &graph_proto.initializer {
            let name = init.name.clone().unwrap_or_default();
            debug!("Loading initializer: {}", name);
            let tensor = tensor_from_proto(init)?;
            initializers.insert(name, tensor);
        }

        // Parse inputs
        let mut inputs = Vec::new();
        for input in &graph_proto.input {
            let name = input.name.clone().unwrap_or_default();
            let shape = shape_from_value_info(input)?;
            debug!("Input: {} with shape {:?}", name, shape);

            // Skip if it's an initializer (weights)
            if !initializers.contains_key(&name) {
                inputs.push(TensorInfo {
                    name: name.clone(),
                    shape,
                    data_type: ronn_core::types::DataType::F32, // Default
                });
            }
        }

        // Parse outputs
        let mut outputs = Vec::new();
        for output in &graph_proto.output {
            let name = output.name.clone().unwrap_or_default();
            let shape = shape_from_value_info(output)?;
            debug!("Output: {} with shape {:?}", name, shape);
            outputs.push(TensorInfo {
                name: name.clone(),
                shape,
                data_type: ronn_core::types::DataType::F32, // Default
            });
        }

        // Parse nodes and build graph
        let mut nodes = Vec::new();
        for (idx, node_proto) in graph_proto.node.iter().enumerate() {
            let op_type = node_proto.op_type.clone().unwrap_or_default();
            let name = node_proto
                .name
                .clone()
                .unwrap_or_else(|| format!("node_{}", idx));

            debug!("Processing node: {} ({})", name, op_type);

            // Parse attributes
            let mut attributes = HashMap::new();
            for attr in &node_proto.attribute {
                let attr_name = attr.name.clone().unwrap_or_default();
                let attr_value = Self::convert_attribute(attr)?;
                attributes.insert(attr_name, attr_value);
            }

            let node = GraphNode {
                id: 0, // Will be assigned by graph.add_node()
                op_type,
                inputs: node_proto.input.clone(),
                outputs: node_proto.output.clone(),
                attributes,
                name: Some(name),
            };
            nodes.push(node);
        }

        // Build the model graph
        let graph = ModelGraph::from_nodes(nodes);

        Ok(LoadedModel {
            graph,
            inputs,
            outputs,
            initializers,
            producer_name: model_proto.producer_name,
            ir_version: model_proto.ir_version.unwrap_or(0),
        })
    }

    /// Convert ONNX AttributeProto to RONN NodeAttribute
    fn convert_attribute(attr: &proto::AttributeProto) -> Result<NodeAttribute> {
        use proto::attribute_proto::AttributeType;

        let attr_type = attr.r#type.unwrap_or(0);

        match attr_type {
            x if x == AttributeType::Float as i32 => {
                Ok(NodeAttribute::Float(attr.f.unwrap_or(0.0) as f64))
            }
            x if x == AttributeType::Int as i32 => {
                Ok(NodeAttribute::Int(attr.i.unwrap_or(0)))
            }
            x if x == AttributeType::String as i32 => {
                let s = attr.s.as_ref()
                    .map(|bytes| String::from_utf8_lossy(bytes).to_string())
                    .unwrap_or_default();
                Ok(NodeAttribute::String(s))
            }
            x if x == AttributeType::Tensor as i32 => {
                if let Some(ref t) = attr.t {
                    // For now, just use empty bytes as a placeholder
                    // Full implementation would serialize the tensor
                    let _tensor = tensor_from_proto(t)?;
                    Ok(NodeAttribute::Tensor(Vec::new()))
                } else {
                    Err(OnnxError::InvalidAttribute {
                        name: attr.name.clone().unwrap_or_default(),
                        reason: "Tensor attribute has no value".to_string(),
                    })
                }
            }
            x if x == AttributeType::Floats as i32 => {
                let floats: Vec<f64> = attr.floats.iter().map(|&f| f as f64).collect();
                Ok(NodeAttribute::FloatArray(floats))
            }
            x if x == AttributeType::Ints as i32 => {
                Ok(NodeAttribute::IntArray(attr.ints.clone()))
            }
            _ => {
                warn!("Unsupported attribute type: {}", attr_type);
                Ok(NodeAttribute::String(format!("unsupported_type_{}", attr_type)))
            }
        }
    }
}

/// Information about a tensor (input/output)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: ronn_core::types::DataType,
}

/// A loaded ONNX model converted to RONN representation
pub struct LoadedModel {
    pub graph: ModelGraph,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    pub initializers: HashMap<String, ronn_core::tensor::Tensor>,
    pub producer_name: Option<String>,
    pub ir_version: i64,
}

impl LoadedModel {
    /// Get the model graph
    pub fn graph(&self) -> &ModelGraph {
        &self.graph
    }

    /// Get input tensor information
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Get output tensor information
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Get initializer tensors (weights, constants)
    pub fn initializers(&self) -> &HashMap<String, ronn_core::tensor::Tensor> {
        &self.initializers
    }
}
