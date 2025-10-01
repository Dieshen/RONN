use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("ONNX error: {0}")]
    OnnxError(#[from] ronn_onnx::OnnxError),

    #[error("Core error: {0}")]
    CoreError(#[from] ronn_core::error::CoreError),

    #[error("Optimization error: {0}")]
    OptimizationError(#[from] ronn_graph::OptimizationError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
