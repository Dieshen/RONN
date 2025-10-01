// Simplified ONNX protobuf types for MVP
// For production, use prost-generated types from official ONNX proto files

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProto {
    pub ir_version: Option<i64>,
    pub producer_name: Option<String>,
    pub graph: Option<GraphProto>,
    pub opset_import_version: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphProto {
    pub node: Vec<NodeProto>,
    pub name: Option<String>,
    pub initializer: Vec<TensorProto>,
    pub input: Vec<ValueInfoProto>,
    pub output: Vec<ValueInfoProto>,
    pub value_info: Vec<ValueInfoProto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProto {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub name: Option<String>,
    pub op_type: Option<String>,
    pub attribute: Vec<AttributeProto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeProto {
    pub name: Option<String>,
    pub f: Option<f32>,
    pub i: Option<i64>,
    pub s: Option<Vec<u8>>,
    pub t: Option<Box<TensorProto>>,
    pub g: Option<Box<GraphProto>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<Vec<u8>>,
    pub tensors: Vec<TensorProto>,
    pub graphs: Vec<GraphProto>,
    pub r#type: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueInfoProto {
    pub name: Option<String>,
    pub r#type: Option<TypeProto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeProto {
    pub value: Option<TypeProtoValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeProtoValue {
    #[serde(rename = "tensor_type")]
    TensorType(TypeProtoTensor),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeProtoTensor {
    pub elem_type: Option<i32>,
    pub shape: Option<TensorShapeProto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShapeProto {
    pub dim: Vec<TensorShapeProtoDimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShapeProtoDimension {
    pub value: Option<DimensionValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionValue {
    #[serde(rename = "dim_value")]
    DimValue(i64),
    #[serde(rename = "dim_param")]
    DimParam(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorProto {
    pub dims: Vec<i64>,
    pub data_type: Option<i32>,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub int64_data: Vec<i64>,
    pub raw_data: Vec<u8>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
    pub name: Option<String>,
}

pub mod tensor_proto {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DataType {
        Undefined = 0,
        Float = 1,
        Uint8 = 2,
        Int8 = 3,
        Uint16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        Uint32 = 12,
        Uint64 = 13,
        Complex64 = 14,
        Complex128 = 15,
        Bfloat16 = 16,
    }
}

pub mod attribute_proto {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AttributeType {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
    }
}

pub mod type_proto {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Value {
        #[serde(rename = "tensor_type")]
        TensorType(Tensor),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Tensor {
        pub elem_type: Option<i32>,
        pub shape: Option<TensorShapeProto>,
    }
}

pub mod tensor_shape_proto {
    use super::*;

    pub mod dimension {
        use super::*;

        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum Value {
            #[serde(rename = "dim_value")]
            DimValue(i64),
            #[serde(rename = "dim_param")]
            DimParam(String),
        }
    }
}
