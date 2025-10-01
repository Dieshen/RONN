use thiserror::Error;

pub type Result<T> = std::result::Result<T, OnnxError>;

#[derive(Error, Debug)]
pub enum OnnxError {
    #[error("Failed to parse ONNX model: {0}")]
    ParseError(String),

    #[error("Unsupported operator: {op_type}")]
    UnsupportedOperator { op_type: String },

    #[error("Invalid attribute: {name} - {reason}")]
    InvalidAttribute { name: String, reason: String },

    #[error("Shape inference failed: {0}")]
    ShapeInferenceError(String),

    #[error("Type conversion error: {0}")]
    TypeConversionError(String),

    #[error("Missing required input: {0}")]
    MissingInput(String),

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Core runtime error: {0}")]
    CoreError(#[from] ronn_core::CoreError),

    #[error("Anyhow error: {0}")]
    AnyhowError(#[from] anyhow::Error),
}
