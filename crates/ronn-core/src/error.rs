// ! Error types for RONN core operations

use thiserror::Error;

/// Result type for core operations
pub type Result<T> = std::result::Result<T, CoreError>;

/// Core error types
#[derive(Error, Debug)]
pub enum CoreError {
    /// Tensor operation failed
    #[error("Tensor operation failed: {0}")]
    TensorError(String),

    /// Shape mismatch between tensors
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Invalid operation attempted
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Graph-related error
    #[error("Graph error: {0}")]
    GraphError(String),

    /// Session-related error
    #[error("Session error: {0}")]
    SessionError(String),

    /// Candle tensor library error
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
