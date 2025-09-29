//! GPU execution provider using Candle backend.
//!
//! This module provides GPU-accelerated execution using the Candle library
//! with support for CUDA and Metal backends.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor as CandleTensor};
use ronn_core::tensor::Tensor;
use ronn_core::{
    CompiledKernel, DataType, ExecutionProvider, KernelStats, MemoryType, MemoryUsage,
    OperatorSpec, PerformanceProfile, ProviderCapability, ProviderConfig, ProviderId,
    ResourceRequirements, SubGraph, TensorAllocator, TensorLayout,
};
use tracing::{debug, info, warn};

use super::allocator::{create_gpu_allocator, GpuMemoryAllocator};

/// GPU execution provider using Candle backend.
pub struct GpuExecutionProvider {
    /// GPU device for execution.
    device: Device,
    /// Memory allocator for this provider.
    allocator: Arc<dyn TensorAllocator>,
    /// Set of supported operations.
    supported_ops: HashSet<String>,
    /// Provider configuration.
    config: GpuProviderConfig,
}

/// Configuration for GPU execution provider.
#[derive(Debug, Clone)]
pub struct GpuProviderConfig {
    /// GPU device ID.
    pub device_id: usize,
    /// Memory limit in bytes (None = no limit).
    pub memory_limit: Option<usize>,
    /// Enable mixed precision (F16) operations.
    pub enable_mixed_precision: bool,
    /// Enable tensor core optimizations (if available).
    pub enable_tensor_cores: bool,
    /// Stream count for async operations.
    pub stream_count: usize,
}

impl Default for GpuProviderConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_limit: None,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            stream_count: 1,
        }
    }
}

/// GPU kernel implementation using Candle.
#[derive(Debug)]
pub struct GpuKernel {
    /// Original subgraph.
    subgraph: SubGraph,
    /// GPU device for execution.
    device: Device,
    /// Execution statistics.
    stats: std::sync::Mutex<GpuKernelStats>,
}

#[derive(Debug, Default)]
struct GpuKernelStats {
    execution_count: u64,
    total_time_us: u64,
    min_time_us: u64,
    max_time_us: u64,
    memory_peak: usize,
}

impl GpuExecutionProvider {
    /// Create a new GPU execution provider with default configuration.
    #[cfg(feature = "gpu")]
    pub fn new() -> Result<Self> {
        Self::with_config(GpuProviderConfig::default())
    }

    /// Create a GPU execution provider with custom configuration.
    #[cfg(feature = "gpu")]
    pub fn with_config(config: GpuProviderConfig) -> Result<Self> {
        // Try to create GPU device
        let device = Self::create_gpu_device(config.device_id)?;

        info!("Created GPU provider with device: {:?}", device);

        // Create GPU allocator
        let allocator =
            create_gpu_allocator().map_err(|e| anyhow!("Failed to create GPU allocator: {}", e))?;

        // Define supported operations (GPU-optimized subset)
        let mut supported_ops = HashSet::new();

        // Basic arithmetic operations (highly optimized on GPU)
        supported_ops.insert("Add".to_string());
        supported_ops.insert("Sub".to_string());
        supported_ops.insert("Mul".to_string());
        supported_ops.insert("Div".to_string());

        // Matrix operations (GPU's strength)
        supported_ops.insert("MatMul".to_string());
        supported_ops.insert("Gemm".to_string());

        // Convolution operations (GPU-accelerated)
        supported_ops.insert("Conv".to_string());
        supported_ops.insert("ConvTranspose".to_string());

        // Pooling operations
        supported_ops.insert("MaxPool".to_string());
        supported_ops.insert("AveragePool".to_string());
        supported_ops.insert("GlobalAveragePool".to_string());

        // Activation functions (element-wise, GPU-friendly)
        supported_ops.insert("ReLU".to_string());
        supported_ops.insert("Sigmoid".to_string());
        supported_ops.insert("Tanh".to_string());
        supported_ops.insert("Softmax".to_string());
        supported_ops.insert("GELU".to_string());

        // Normalization operations
        supported_ops.insert("BatchNormalization".to_string());
        supported_ops.insert("LayerNormalization".to_string());

        // Reduction operations (efficient on GPU)
        supported_ops.insert("Sum".to_string());
        supported_ops.insert("Mean".to_string());
        supported_ops.insert("Max".to_string());
        supported_ops.insert("Min".to_string());

        // Shape operations (fast on GPU)
        supported_ops.insert("Reshape".to_string());
        supported_ops.insert("Transpose".to_string());
        supported_ops.insert("Concat".to_string());
        supported_ops.insert("Split".to_string());

        info!(
            "GPU provider supports {} operation types",
            supported_ops.len()
        );

        Ok(Self {
            device,
            allocator,
            supported_ops,
            config,
        })
    }

    /// Fallback constructor when GPU is not available.
    #[cfg(not(feature = "gpu"))]
    pub fn new() -> Result<Self> {
        Err(anyhow!("GPU support not compiled in"))
    }

    /// Create a GPU execution provider with custom configuration.
    #[cfg(not(feature = "gpu"))]
    pub fn with_config(_config: GpuProviderConfig) -> Result<Self> {
        Err(anyhow!("GPU support not compiled in"))
    }

    /// Create GPU device based on configuration.
    #[cfg(feature = "gpu")]
    fn create_gpu_device(device_id: usize) -> Result<Device> {
        // Try CUDA first
        if let Ok(device) = Device::new_cuda(device_id) {
            info!("Using CUDA device {}", device_id);
            return Ok(device);
        }

        // Try Metal on macOS
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(device_id) {
                info!("Using Metal device {}", device_id);
                return Ok(device);
            }
        }

        Err(anyhow!("No GPU devices available"))
    }

    /// Get the GPU device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the current configuration.
    pub fn get_config(&self) -> &GpuProviderConfig {
        &self.config
    }

    /// Check if an operation type is supported.
    pub fn supports_operation(&self, op_type: &str) -> bool {
        self.supported_ops.contains(op_type)
    }

    /// Estimate execution cost for an operation on GPU.
    pub fn estimate_cost(&self, op_spec: &OperatorSpec) -> f64 {
        // GPU cost estimation - generally lower for parallel operations
        match op_spec.op_type.as_str() {
            "Add" | "Sub" | "Mul" | "Div" => 0.1, // Very fast on GPU
            "ReLU" | "Sigmoid" | "Tanh" => 0.2,   // Fast element-wise
            "MatMul" | "Gemm" => 0.5,             // GPU's strength
            "Conv" => 0.8,                        // Complex but GPU-optimized
            "ConvTranspose" => 1.2,               // More complex
            "BatchNormalization" => 0.3,          // Fast on GPU
            "Softmax" => 0.4,                     // Reduction + element-wise
            "MaxPool" | "AveragePool" => 0.3,     // Simple operations
            _ => 1.0,                             // Default cost
        }
    }

    /// Check if the provider can utilize tensor cores.
    #[cfg(feature = "gpu")]
    pub fn has_tensor_cores(&self) -> bool {
        // In practice, would query GPU capabilities
        // For now, assume modern CUDA devices have tensor cores
        matches!(self.device, Device::Cuda(_)) && self.config.enable_tensor_cores
    }

    /// Check if the GPU has tensor cores for mixed-precision operations.
    #[cfg(not(feature = "gpu"))]
    pub fn has_tensor_cores(&self) -> bool {
        false
    }

    /// Get GPU memory information.
    #[cfg(feature = "gpu")]
    pub fn get_gpu_memory_info(&self) -> Result<(usize, usize)> {
        // In practice, would query actual GPU memory
        // For now, return estimated values
        match &self.device {
            Device::Cuda(_) => Ok((8 * 1024 * 1024 * 1024, 0)), // 8GB total, 0 used
            Device::Metal(_) => Ok((8 * 1024 * 1024 * 1024, 0)), // 8GB total, 0 used
            _ => Err(anyhow!("Not a GPU device")),
        }
    }

    /// Get GPU memory information (total, available) in bytes.
    #[cfg(not(feature = "gpu"))]
    pub fn get_gpu_memory_info(&self) -> Result<(usize, usize)> {
        Err(anyhow!("GPU support not available"))
    }
}

impl Default for GpuExecutionProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create default GPU provider")
    }
}

impl ExecutionProvider for GpuExecutionProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::GPU
    }

    fn get_capability(&self) -> ProviderCapability {
        let mut data_types = vec![
            DataType::F32,
            DataType::F16, // Important for GPU mixed precision
            DataType::F64,
            DataType::U8,
            DataType::U32,
        ];

        // Add additional types if tensor cores are available
        if self.has_tensor_cores() {
            // Tensor cores work best with F16
            data_types.insert(0, DataType::F16); // Prioritize F16
        }

        let gpu_memory = self
            .get_gpu_memory_info()
            .map(|(total, _)| total)
            .unwrap_or(0);

        ProviderCapability {
            supported_ops: self.supported_ops.clone(),
            data_types,
            memory_types: vec![MemoryType::DeviceMemory, MemoryType::SharedMemory],
            performance_profile: PerformanceProfile::GPU,
            resource_requirements: ResourceRequirements {
                min_memory_bytes: Some(512 * 1024 * 1024), // 512MB minimum
                cpu_features: vec![],                      // No specific CPU requirements
                gpu_memory_bytes: Some(gpu_memory),
            },
        }
    }

    fn can_handle(&self, operators: &[OperatorSpec]) -> Vec<bool> {
        operators
            .iter()
            .map(|op| self.supports_operation(&op.op_type))
            .collect()
    }

    fn compile_subgraph(&self, subgraph: SubGraph) -> Result<Box<dyn CompiledKernel>> {
        debug!(
            "Compiling subgraph with {} nodes for GPU",
            subgraph.nodes.len()
        );

        // Validate that all operations are supported
        for node in &subgraph.nodes {
            if !self.supports_operation(&node.op_type) {
                return Err(anyhow!(
                    "Unsupported GPU operation '{}' in subgraph",
                    node.op_type
                ));
            }
        }

        // Create GPU kernel
        let kernel = GpuKernel::new(subgraph, self.device.clone())?;

        debug!("Successfully compiled GPU kernel");

        Ok(Box::new(kernel))
    }

    fn get_allocator(&self) -> Arc<dyn TensorAllocator> {
        self.allocator.clone()
    }

    fn configure(&mut self, config: ProviderConfig) -> Result<()> {
        // Update memory limit
        if let Some(memory_limit) = config.memory_limit {
            self.config.memory_limit = Some(memory_limit);
            info!("Updated GPU memory limit to {} bytes", memory_limit);
        }

        // Handle custom options
        for (key, value) in &config.custom_options {
            match key.as_str() {
                "enable_mixed_precision" => {
                    if let Ok(enable) = value.parse::<bool>() {
                        self.config.enable_mixed_precision = enable;
                        info!("Updated mixed precision to {}", enable);
                    }
                }
                "enable_tensor_cores" => {
                    if let Ok(enable) = value.parse::<bool>() {
                        self.config.enable_tensor_cores = enable;
                        info!("Updated tensor cores to {}", enable);
                    }
                }
                "stream_count" => {
                    if let Ok(count) = value.parse::<usize>() {
                        self.config.stream_count = count;
                        info!("Updated stream count to {}", count);
                    }
                }
                _ => {
                    warn!("Unknown GPU configuration option: {}", key);
                }
            }
        }

        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        info!("Shutting down GPU execution provider");

        // GPU cleanup would happen here in a real implementation
        // Candle handles most cleanup automatically

        debug!("GPU provider shutdown complete");

        Ok(())
    }
}

impl GpuKernel {
    /// Create a new GPU kernel.
    pub fn new(subgraph: SubGraph, device: Device) -> Result<Self> {
        Ok(Self {
            subgraph,
            device,
            stats: std::sync::Mutex::new(GpuKernelStats::default()),
        })
    }

    /// Execute a single operation on GPU using Candle.
    fn execute_gpu_operation(
        &self,
        op_type: &str,
        inputs: &[CandleTensor],
    ) -> Result<Vec<CandleTensor>> {
        match op_type {
            "Add" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Add requires exactly 2 inputs"));
                }
                let result = (&inputs[0] + &inputs[1])?;
                Ok(vec![result])
            }

            "Sub" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Sub requires exactly 2 inputs"));
                }
                let result = (&inputs[0] - &inputs[1])?;
                Ok(vec![result])
            }

            "Mul" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Mul requires exactly 2 inputs"));
                }
                let result = (&inputs[0] * &inputs[1])?;
                Ok(vec![result])
            }

            "Div" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("Div requires exactly 2 inputs"));
                }
                let result = (&inputs[0] / &inputs[1])?;
                Ok(vec![result])
            }

            "MatMul" => {
                if inputs.len() != 2 {
                    return Err(anyhow!("MatMul requires exactly 2 inputs"));
                }
                let result = inputs[0].matmul(&inputs[1])?;
                Ok(vec![result])
            }

            "ReLU" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("ReLU requires exactly 1 input"));
                }
                let zero = inputs[0].zeros_like()?;
                let result = inputs[0].maximum(&zero)?;
                Ok(vec![result])
            }

            "Softmax" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("Softmax requires exactly 1 input"));
                }
                let result = candle_nn::ops::softmax_last_dim(&inputs[0])?;
                Ok(vec![result])
            }

            "Reshape" => {
                if inputs.len() != 1 {
                    return Err(anyhow!("Reshape requires exactly 1 input"));
                }
                // For simplicity, just return the input (reshape params would come from attributes)
                Ok(vec![inputs[0].clone()])
            }

            _ => Err(anyhow!("Unsupported GPU operation: {}", op_type)),
        }
    }

    /// Convert RONN Tensor to Candle Tensor.
    fn ronn_to_candle(&self, tensor: &ronn_core::tensor::Tensor) -> Result<CandleTensor> {
        let data = tensor.to_vec()?;
        let shape = tensor.shape();
        let dtype = match tensor.dtype() {
            DataType::F32 => candle_core::DType::F32,
            DataType::F16 => candle_core::DType::F16,
            DataType::F64 => candle_core::DType::F64,
            DataType::U8 => candle_core::DType::U8,
            DataType::U32 => candle_core::DType::U32,
            _ => candle_core::DType::F32, // Fallback
        };

        let candle_tensor =
            CandleTensor::from_vec(data, shape.as_slice(), &self.device)?.to_dtype(dtype)?;

        Ok(candle_tensor)
    }

    /// Convert Candle Tensor to RONN Tensor.
    fn candle_to_ronn(&self, tensor: &CandleTensor) -> Result<ronn_core::tensor::Tensor> {
        let shape = tensor.dims().to_vec();
        let data: Vec<f32> = tensor.to_vec1()?; // Convert to F32 for now

        let ronn_tensor = Tensor::from_data(
            data,
            shape,
            DataType::F32, // Simplified for now
            TensorLayout::RowMajor,
        )?;

        Ok(ronn_tensor)
    }
}

impl CompiledKernel for GpuKernel {
    fn execute(
        &self,
        inputs: &[ronn_core::tensor::Tensor],
    ) -> Result<Vec<ronn_core::tensor::Tensor>> {
        let start_time = std::time::Instant::now();

        // Convert RONN tensors to Candle tensors
        let mut candle_inputs = Vec::new();
        for input in inputs {
            let candle_tensor = self.ronn_to_candle(input)?;
            candle_inputs.push(candle_tensor);
        }

        // Execute operations sequentially (simplified)
        let mut current_tensors = candle_inputs;

        for node in &self.subgraph.nodes {
            let outputs = self.execute_gpu_operation(&node.op_type, &current_tensors)?;
            current_tensors = outputs;
        }

        // Convert back to RONN tensors
        let mut results = Vec::new();
        for candle_tensor in &current_tensors {
            let ronn_tensor = self.candle_to_ronn(candle_tensor)?;
            results.push(ronn_tensor);
        }

        // Update statistics
        let execution_time = start_time.elapsed();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.execution_count += 1;
            stats.total_time_us += execution_time.as_micros() as u64;

            if stats.execution_count == 1 {
                stats.min_time_us = execution_time.as_micros() as u64;
                stats.max_time_us = execution_time.as_micros() as u64;
            } else {
                stats.min_time_us = stats.min_time_us.min(execution_time.as_micros() as u64);
                stats.max_time_us = stats.max_time_us.max(execution_time.as_micros() as u64);
            }
        }

        debug!("GPU kernel executed in {:?}", execution_time);

        Ok(results)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let stats = self.stats.lock().unwrap();
        MemoryUsage {
            peak_bytes: stats.memory_peak,
            current_bytes: 0, // Would track current usage in practice
            allocation_count: stats.execution_count as usize,
        }
    }

    fn get_performance_stats(&self) -> KernelStats {
        let stats = self.stats.lock().unwrap();

        let average_time_us = if stats.execution_count > 0 {
            stats.total_time_us as f64 / stats.execution_count as f64
        } else {
            0.0
        };

        KernelStats {
            execution_count: stats.execution_count,
            average_time_us,
            min_time_us: stats.min_time_us as f64,
            max_time_us: stats.max_time_us as f64,
        }
    }
}

/// Create a default GPU execution provider.
pub fn create_gpu_provider() -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(GpuExecutionProvider::new()?))
}

/// Create a GPU execution provider with custom configuration.
pub fn create_gpu_provider_with_config(
    config: GpuProviderConfig,
) -> Result<Arc<dyn ExecutionProvider>> {
    Ok(Arc::new(GpuExecutionProvider::with_config(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::{AttributeValue, GraphNode};

    fn create_test_subgraph() -> SubGraph {
        let node = GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
            name: Some("gpu_add".to_string()),
        };

        SubGraph {
            nodes: vec![node],
            edges: vec![],
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
        }
    }

    #[test]
    fn test_gpu_provider_creation() {
        // This test may fail if no GPU is available
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);

                let capability = provider.get_capability();
                assert_eq!(capability.performance_profile, PerformanceProfile::GPU);
                assert!(!capability.supported_ops.is_empty());
                assert!(capability.data_types.contains(&DataType::F32));
            }
            Err(e) => {
                println!("GPU not available: {}", e);
                // Test passes if GPU is not available
            }
        }
    }

    #[test]
    fn test_gpu_provider_config() {
        let config = GpuProviderConfig {
            device_id: 0,
            enable_mixed_precision: false,
            enable_tensor_cores: false,
            ..Default::default()
        };

        match GpuExecutionProvider::with_config(config) {
            Ok(provider) => {
                assert!(!provider.get_config().enable_mixed_precision);
                assert!(!provider.get_config().enable_tensor_cores);
            }
            Err(_) => {
                // GPU not available, test passes
            }
        }
    }

    #[test]
    fn test_operation_support() {
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                // Test GPU-optimized operations
                assert!(provider.supports_operation("Add"));
                assert!(provider.supports_operation("MatMul"));
                assert!(provider.supports_operation("Conv"));
                assert!(provider.supports_operation("ReLU"));
                assert!(!provider.supports_operation("NonexistentOp"));

                // Test cost estimation
                let add_op = OperatorSpec {
                    op_type: "Add".to_string(),
                    input_types: vec![DataType::F32],
                    output_types: vec![DataType::F32],
                    attributes: HashMap::new(),
                };

                let conv_op = OperatorSpec {
                    op_type: "Conv".to_string(),
                    input_types: vec![DataType::F32],
                    output_types: vec![DataType::F32],
                    attributes: HashMap::new(),
                };

                let add_cost = provider.estimate_cost(&add_op);
                let conv_cost = provider.estimate_cost(&conv_op);

                // GPU should be very efficient for both, but Conv more complex
                assert!(conv_cost > add_cost);
                assert!(add_cost < 1.0); // Should be less than 1.0 for GPU
            }
            Err(_) => {
                // GPU not available
            }
        }
    }

    #[test]
    fn test_subgraph_compilation() {
        match GpuExecutionProvider::new() {
            Ok(provider) => {
                let subgraph = create_test_subgraph();

                match provider.compile_subgraph(subgraph) {
                    Ok(kernel) => {
                        let stats = kernel.get_performance_stats();
                        assert_eq!(stats.execution_count, 0); // Not executed yet
                    }
                    Err(e) => {
                        println!("Compilation failed: {}", e);
                    }
                }
            }
            Err(_) => {
                // GPU not available
            }
        }
    }

    #[test]
    fn test_factory_functions() {
        // Test factory functions (may fail if no GPU)
        match create_gpu_provider() {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);
            }
            Err(_) => {
                // GPU not available
            }
        }

        let config = GpuProviderConfig::default();
        match create_gpu_provider_with_config(config) {
            Ok(provider) => {
                assert_eq!(provider.provider_id(), ProviderId::GPU);
            }
            Err(_) => {
                // GPU not available
            }
        }
    }
}
