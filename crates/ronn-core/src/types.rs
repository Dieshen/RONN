//! Core type definitions for the RONN runtime.
//!
//! This module provides the foundational data types used throughout the RONN
//! runtime, including tensors, model graphs, and execution interfaces.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for nodes in a model graph.
pub type NodeId = usize;

/// Unique identifier for inference sessions.
pub type SessionId = Uuid;

/// Alias for node attributes (same as AttributeValue)
pub type NodeAttribute = AttributeValue;

/// Alias for provider types (same as ProviderId)
pub type ProviderType = ProviderId;

/// Unique identifier for execution providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderId {
    /// CPU execution provider with SIMD optimizations.
    CPU,
    /// GPU execution provider using Candle.
    GPU,
    /// WebAssembly execution provider for browser deployment.
    WebAssembly,
    /// BitNet specialized provider for 1-bit quantized models.
    BitNet,
    /// Custom hardware provider.
    Custom(u32),
}

/// Supported data types for tensors and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point.
    F16,
    /// BFloat16 (Brain Floating Point).
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 32-bit integer.
    U32,
    /// Boolean values.
    Bool,
    /// 64-bit floating point (limited support).
    F64,
}

/// Multi-dimensional tensor with shape and data type information.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Raw data storage.
    pub data: Vec<f32>,
    /// Shape of the tensor (dimensions).
    pub shape: Vec<usize>,
    /// Data type of tensor elements.
    pub dtype: DataType,
    /// Memory layout optimization hint.
    pub layout: TensorLayout,
}

/// Memory layout options for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Row-major (C-style) layout.
    RowMajor,
    /// Column-major (Fortran-style) layout.
    ColumnMajor,
}

/// Attribute values for model graph nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    /// Integer attribute.
    Int(i64),
    /// Float attribute.
    Float(f64),
    /// String attribute.
    String(String),
    /// Boolean attribute.
    Bool(bool),
    /// Array of integers.
    IntArray(Vec<i64>),
    /// Array of floats.
    FloatArray(Vec<f64>),
    /// Array of strings.
    StringArray(Vec<String>),
    /// Tensor attribute.
    Tensor(Vec<u8>), // Serialized tensor data
}

/// Represents a complete model graph with nodes, edges, and metadata.
#[derive(Debug, Clone)]
pub struct ModelGraph {
    /// All nodes in the graph.
    pub nodes: Vec<GraphNode>,
    /// Edges connecting nodes.
    pub edges: Vec<GraphEdge>,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Graph metadata.
    pub metadata: HashMap<String, AttributeValue>,
}

/// A single node in the model graph representing an operation.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// Operation type (e.g., "Conv", "ReLU", "MatMul").
    pub op_type: String,
    /// Operation-specific attributes.
    pub attributes: HashMap<String, AttributeValue>,
    /// Names of input tensors.
    pub inputs: Vec<String>,
    /// Names of output tensors.
    pub outputs: Vec<String>,
    /// Human-readable name for the node.
    pub name: Option<String>,
}

/// An edge in the model graph connecting two nodes.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID.
    pub from_node: NodeId,
    /// Target node ID.
    pub to_node: NodeId,
    /// Name of the tensor being passed.
    pub tensor_name: String,
    /// Shape of the tensor (if known).
    pub tensor_shape: Option<Vec<usize>>,
    /// Data type of the tensor.
    pub tensor_dtype: DataType,
}

/// Represents a compiled subgraph ready for execution.
pub trait CompiledKernel: Send + Sync {
    /// Execute the kernel with the given inputs.
    fn execute(
        &self,
        inputs: &[crate::tensor::Tensor],
    ) -> anyhow::Result<Vec<crate::tensor::Tensor>>;

    /// Get memory usage statistics for this kernel.
    fn get_memory_usage(&self) -> MemoryUsage;

    /// Get performance statistics from recent executions.
    fn get_performance_stats(&self) -> KernelStats;
}

/// Memory usage information.
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Current memory usage in bytes.
    pub current_bytes: usize,
    /// Number of allocations.
    pub allocation_count: usize,
}

/// Performance statistics for kernel execution.
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Total number of executions.
    pub execution_count: u64,
    /// Average execution time in microseconds.
    pub average_time_us: f64,
    /// Minimum execution time in microseconds.
    pub min_time_us: f64,
    /// Maximum execution time in microseconds.
    pub max_time_us: f64,
}

/// Core execution provider interface.
pub trait ExecutionProvider: Send + Sync {
    /// Get the unique identifier for this provider.
    fn provider_id(&self) -> ProviderId;

    /// Get the capabilities of this execution provider.
    fn get_capability(&self) -> ProviderCapability;

    /// Check if this provider can handle specific operations.
    fn can_handle(&self, operators: &[OperatorSpec]) -> Vec<bool>;

    /// Compile a subgraph into an executable kernel.
    fn compile_subgraph(&self, subgraph: SubGraph) -> anyhow::Result<Box<dyn CompiledKernel>>;

    /// Get the memory allocator for this provider.
    fn get_allocator(&self) -> std::sync::Arc<dyn TensorAllocator>;

    /// Configure the provider with specific settings.
    fn configure(&mut self, config: ProviderConfig) -> anyhow::Result<()>;

    /// Shutdown and cleanup provider resources.
    fn shutdown(&self) -> anyhow::Result<()>;
}

/// Capabilities reported by an execution provider.
#[derive(Debug, Clone)]
pub struct ProviderCapability {
    /// Supported operation types.
    pub supported_ops: std::collections::HashSet<String>,
    /// Supported data types.
    pub data_types: Vec<DataType>,
    /// Supported memory types.
    pub memory_types: Vec<MemoryType>,
    /// Performance characteristics.
    pub performance_profile: PerformanceProfile,
    /// Resource requirements.
    pub resource_requirements: ResourceRequirements,
}

/// Memory types supported by providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// System RAM.
    SystemRAM,
    /// GPU device memory.
    DeviceMemory,
    /// Shared memory between CPU and GPU.
    SharedMemory,
    /// Memory-mapped files.
    MappedMemory,
}

/// Performance profile categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceProfile {
    /// Optimized for CPU workloads.
    CPU,
    /// Optimized for GPU workloads.
    GPU,
    /// Optimized for memory-constrained environments.
    MemoryOptimized,
    /// Optimized for low-power/mobile devices.
    PowerEfficient,
}

/// Resource requirements for a provider.
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Minimum memory in bytes.
    pub min_memory_bytes: Option<usize>,
    /// Required CPU features.
    pub cpu_features: Vec<String>,
    /// GPU memory requirements in bytes.
    pub gpu_memory_bytes: Option<usize>,
}

/// Configuration for execution providers.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Number of worker threads.
    pub thread_count: Option<usize>,
    /// Memory limit in bytes.
    pub memory_limit: Option<usize>,
    /// Optimization level.
    pub optimization_level: OptimizationLevel,
    /// Custom configuration options.
    pub custom_options: HashMap<String, String>,
}

/// Optimization levels for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization.
    None,
    /// Basic optimizations.
    Basic,
    /// Aggressive optimizations.
    Aggressive,
    /// Custom optimization flags.
    Custom,
}

/// Memory allocator interface for tensors.
pub trait TensorAllocator: Send + Sync {
    /// Allocate memory for a tensor with the given shape and data type.
    fn allocate(&self, shape: &[usize], dtype: DataType) -> anyhow::Result<TensorBuffer>;

    /// Deallocate a tensor buffer.
    fn deallocate(&self, buffer: TensorBuffer) -> anyhow::Result<()>;

    /// Get memory information for this allocator.
    fn get_memory_info(&self) -> MemoryInfo;
}

/// Buffer for tensor data with metadata.
#[derive(Debug)]
pub struct TensorBuffer {
    /// Pointer to the allocated memory.
    pub ptr: *mut u8,
    /// Size of the allocated memory in bytes.
    pub size: usize,
    /// Memory alignment in bytes.
    pub alignment: usize,
    /// Type of memory.
    pub memory_type: MemoryType,
}

// SAFETY: TensorBuffer is used in controlled allocation contexts where
// the pointer is valid and exclusive access is managed by the allocator
unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}

/// Memory allocator information.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total available memory in bytes.
    pub total_bytes: usize,
    /// Currently allocated memory in bytes.
    pub allocated_bytes: usize,
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
}

/// Specification for an operator.
#[derive(Debug, Clone)]
pub struct OperatorSpec {
    /// Operation type name.
    pub op_type: String,
    /// Input data types.
    pub input_types: Vec<DataType>,
    /// Output data types.
    pub output_types: Vec<DataType>,
    /// Operation attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

/// A subgraph that can be compiled and executed.
#[derive(Debug, Clone)]
pub struct SubGraph {
    /// Nodes in this subgraph.
    pub nodes: Vec<GraphNode>,
    /// Edges within this subgraph.
    pub edges: Vec<GraphEdge>,
    /// Input tensor names for this subgraph.
    pub inputs: Vec<String>,
    /// Output tensor names for this subgraph.
    pub outputs: Vec<String>,
}
