// ONNX compatibility layer for RONN runtime
//
// This crate provides ONNX model loading, operator support, and conversion
// to the internal graph representation.

mod error;
mod generated;
mod loader;
mod ops;
mod proto;
mod types;

pub use error::{OnnxError, Result};
pub use loader::{LoadedModel, ModelLoader};
pub use ops::*;
pub use types::*;

// Generated ONNX protobuf types (via prost)
pub mod onnx_proto {
    pub use crate::generated::*;
}

// Simplified ONNX protobuf types (manual, for backward compatibility)
pub use proto::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_conversion() {
        let onnx_type = proto::tensor_proto::DataType::Float;
        let ronn_type = DataTypeMapper::from_onnx(onnx_type as i32);
        assert!(ronn_type.is_ok());
    }
}
