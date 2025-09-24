//! RONN Core Runtime Engine
//!
//! This crate provides the foundational components of the RONN (Rust ONNX Neural Network)
//! runtime, including tensor operations, model graph representation, and core execution
//! interfaces.
//!
//! ## Architecture
//!
//! The core engine follows a layered architecture:
//! - **Types**: Fundamental data structures for tensors, graphs, and metadata
//! - **Session**: Management of inference sessions and resource isolation
//! - **Tensor**: Multi-dimensional array operations with Candle integration
//! - **Graph**: Model representation and manipulation utilities
//!
//! ## Example
//!
//! ```rust
//! use ronn_core::{Tensor, DataType, TensorLayout};
//!
//! // Create a 2x3 tensor with zeros
//! let data = vec![0.0; 6];
//! let tensor = Tensor {
//!     data,
//!     shape: vec![2, 3],
//!     dtype: DataType::F32,
//!     layout: TensorLayout::RowMajor,
//! };
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

pub mod types;

// Re-export commonly used types
pub use types::{
    AttributeValue, CompiledKernel, DataType, ExecutionProvider, GraphEdge, GraphNode, MemoryType,
    ModelGraph, NodeId, OperatorSpec, PerformanceProfile, ProviderId, ProviderCapability,
    ProviderConfig, ResourceRequirements, SessionId, SubGraph, Tensor, TensorAllocator,
    TensorBuffer, TensorLayout,
};

/// Result type alias for core operations.
pub type Result<T> = anyhow::Result<T>;
