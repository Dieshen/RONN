use thiserror::Error;

pub type Result<T> = std::result::Result<T, OptimizationError>;

#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Optimization pass failed: {pass_name} - {reason}")]
    PassFailed { pass_name: String, reason: String },

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Constant folding error: {0}")]
    ConstantFoldingError(String),

    #[error("Node fusion error: {0}")]
    NodeFusionError(String),

    #[error("Layout optimization error: {0}")]
    LayoutError(String),

    #[error("Core runtime error: {0}")]
    CoreError(#[from] ronn_core::error::CoreError),
}
