# Execution Provider Framework - Detailed Design

## Overview

The Execution Provider (EP) Framework is the hardware abstraction layer of the Rust ML Runtime, inspired by ONNX Runtime's execution provider architecture. It enables the runtime to leverage diverse hardware accelerators while maintaining a unified API and optimal performance across platforms.

## Table of Contents
1. [Design Principles](#design-principles)
2. [Architecture Overview](#architecture-overview)
3. [Core Interfaces](#core-interfaces)
4. [Built-in Providers](#built-in-providers)
5. [Graph Partitioning](#graph-partitioning)
6. [Memory Management](#memory-management)
7. [Performance Optimization](#performance-optimization)
8. [Implementation Details](#implementation-details)

## Design Principles

### 1. Hardware Agnostic Interface
The framework provides a unified interface regardless of underlying hardware, allowing applications to run on any supported platform without modification.

### 2. Performance First
Each execution provider is optimized for its target hardware, leveraging specialized libraries and hardware-specific features for maximum performance.

### 3. Graceful Degradation
The system automatically falls back to alternative providers if preferred hardware is unavailable or if operations are unsupported.

### 4. Extensibility
Third-party providers can be easily integrated through the standardized interface, enabling support for new hardware accelerators.

### 5. Pure Rust Implementation
All built-in providers are implemented in pure Rust, avoiding FFI overhead and maintaining memory safety guarantees.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                Core Inference Engine                    │
├─────────────────────────────────────────────────────────┤
│              Execution Provider Framework               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Provider A  │  │ Provider B  │  │ Provider C  │ ...  │
│  │ (CPU/SIMD)  │  │ (GPU/Candle)│  │ (Custom)    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────────────────────┤
│                  Hardware Layer                         │
│     CPU          GPU          NPU         Custom        │
└─────────────────────────────────────────────────────────┘
```

### Provider Registration and Discovery

```rust
pub struct ExecutionProviderRegistry {
    providers: BTreeMap<ProviderId, Arc<dyn ExecutionProvider>>,
    provider_order: Vec<ProviderId>,
    capability_cache: DashMap<ModelHash, Vec<ProviderCapability>>,
}

impl ExecutionProviderRegistry {
    pub fn register<P: ExecutionProvider + 'static>(&mut self, provider: P) {
        let id = provider.provider_id();
        let provider = Arc::new(provider);
        self.providers.insert(id, provider.clone());
        self.update_provider_order(id);
    }
    
    pub fn get_available_providers(&self) -> Vec<ProviderId> {
        self.provider_order.clone()
    }
    
    pub fn partition_graph(&self, graph: &ModelGraph) -> GraphPartition {
        GraphPartitioner::new(&self.providers).partition(graph)
    }
}
```

## Core Interfaces

### ExecutionProvider Trait

The primary interface that all execution providers must implement:

```rust
pub trait ExecutionProvider: Send + Sync {
    /// Unique identifier for this provider
    fn provider_id(&self) -> ProviderId;
    
    /// Hardware requirements and capabilities
    fn get_capability(&self) -> ProviderCapability;
    
    /// Check if provider can handle specific operations
    fn can_handle(&self, operators: &[OperatorSpec]) -> Vec<bool>;
    
    /// Compile a subgraph into an executable kernel
    fn compile_subgraph(&self, subgraph: SubGraph) -> Result<Box<dyn CompiledKernel>>;
    
    /// Get the memory allocator for this provider
    fn get_allocator(&self) -> Arc<dyn TensorAllocator>;
    
    /// Provider-specific configuration
    fn configure(&mut self, config: ProviderConfig) -> Result<()>;
    
    /// Cleanup resources
    fn shutdown(&self) -> Result<()>;
}

pub struct ProviderCapability {
    pub supported_ops: HashSet<OperatorType>,
    pub data_types: Vec<DataType>,
    pub memory_types: Vec<MemoryType>,
    pub performance_characteristics: PerformanceProfile,
    pub resource_requirements: ResourceRequirements,
}

pub trait CompiledKernel: Send + Sync {
    fn execute(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
    fn get_memory_usage(&self) -> MemoryUsage;
    fn get_performance_stats(&self) -> KernelStats;
}
```

### Memory Allocator Interface

Each provider manages its own memory allocation strategy:

```rust
pub trait TensorAllocator: Send + Sync {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer>;
    fn deallocate(&self, buffer: TensorBuffer) -> Result<()>;
    fn get_memory_info(&self) -> MemoryInfo;
}

pub struct TensorBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    pub alignment: usize,
    pub memory_type: MemoryType,
}

pub enum MemoryType {
    SystemRAM,
    DeviceMemory,
    SharedMemory,
    MappedMemory,
}
```

## Built-in Providers

### 1. CPU Execution Provider

Optimized for multi-core CPUs with SIMD acceleration:

```rust
pub struct CPUExecutionProvider {
    thread_pool: ThreadPool,
    simd_features: SIMDFeatures,
    numa_topology: NumaTopology,
    allocator: Arc<CPUAllocator>,
}

impl CPUExecutionProvider {
    pub fn new() -> Result<Self> {
        let thread_count = std::thread::available_parallelism()?.get();
        let thread_pool = ThreadPool::new(thread_count)?;
        
        Ok(Self {
            thread_pool,
            simd_features: detect_simd_features(),
            numa_topology: NumaTopology::detect()?,
            allocator: Arc::new(CPUAllocator::new()),
        })
    }
    
    fn execute_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match self.simd_features {
            SIMDFeatures::AVX512 => self.matmul_avx512(a, b),
            SIMDFeatures::AVX2 => self.matmul_avx2(a, b),
            SIMDFeatures::NEON => self.matmul_neon(a, b),
            _ => self.matmul_fallback(a, b),
        }
    }
}

impl ExecutionProvider for CPUExecutionProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::CPU
    }
    
    fn get_capability(&self) -> ProviderCapability {
        ProviderCapability {
            supported_ops: cpu_supported_ops(),
            data_types: vec![DataType::F32, DataType::F16, DataType::I8],
            memory_types: vec![MemoryType::SystemRAM],
            performance_characteristics: PerformanceProfile::CPU,
            resource_requirements: ResourceRequirements::minimal(),
        }
    }
    
    fn compile_subgraph(&self, subgraph: SubGraph) -> Result<Box<dyn CompiledKernel>> {
        let optimized_graph = self.optimize_for_cpu(subgraph)?;
        Ok(Box::new(CPUCompiledKernel::new(optimized_graph, &self.thread_pool)))
    }
}
```

### 2. GPU Execution Provider

Leveraging Candle for GPU acceleration:

```rust
pub struct GPUExecutionProvider {
    device: candle_core::Device,
    memory_pool: GpuMemoryPool,
    stream_manager: StreamManager,
    allocator: Arc<GPUAllocator>,
}

impl GPUExecutionProvider {
    pub fn new() -> Result<Self> {
        let device = candle_core::Device::new_cuda(0)?;
        
        Ok(Self {
            device,
            memory_pool: GpuMemoryPool::new()?,
            stream_manager: StreamManager::new()?,
            allocator: Arc::new(GPUAllocator::new(device.clone())),
        })
    }
    
    fn execute_conv2d(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let candle_input = self.to_candle_tensor(input)?;
        let candle_weight = self.to_candle_tensor(weight)?;
        
        let result = candle_input.conv2d(&candle_weight)?;
        self.from_candle_tensor(result)
    }
}

impl ExecutionProvider for GPUExecutionProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::GPU
    }
    
    fn get_capability(&self) -> ProviderCapability {
        ProviderCapability {
            supported_ops: gpu_supported_ops(),
            data_types: vec![DataType::F32, DataType::F16, DataType::BF16],
            memory_types: vec![MemoryType::DeviceMemory],
            performance_characteristics: PerformanceProfile::GPU,
            resource_requirements: ResourceRequirements::gpu_memory(4_000_000_000), // 4GB
        }
    }
}
```

### 3. BitNet Execution Provider

Specialized provider for 1-bit quantized models:

```rust
pub struct BitNetExecutionProvider {
    cpu_provider: CPUExecutionProvider,
    quantization_cache: DashMap<TensorId, BitNetTensor>,
}

impl BitNetExecutionProvider {
    fn execute_bitnet_linear(&self, input: &Tensor, weight: &BitNetTensor) -> Result<Tensor> {
        // Specialized 1-bit matrix multiplication
        let binary_weight = &weight.binary_weights;
        let scale_factor = weight.scale_factor;
        
        let result = self.binary_matmul(input, binary_weight)?;
        Ok(result * scale_factor)
    }
    
    fn binary_matmul(&self, input: &Tensor, binary_weight: &Tensor) -> Result<Tensor> {
        // Optimized binary operations using bit manipulation
        // and SIMD instructions for extreme efficiency
        unimplemented!("BitNet-specific optimizations")
    }
}
```

## Graph Partitioning

The graph partitioner analyzes the model graph and assigns operators to the most suitable execution providers:

```rust
pub struct GraphPartitioner {
    providers: BTreeMap<ProviderId, Arc<dyn ExecutionProvider>>,
    cost_estimator: CostEstimator,
}

impl GraphPartitioner {
    pub fn partition(&self, graph: &ModelGraph) -> GraphPartition {
        let mut partition = GraphPartition::new();
        
        // Phase 1: Capability-based assignment
        for node in graph.topological_order() {
            let candidates = self.find_capable_providers(node);
            let selected = self.select_optimal_provider(&candidates, node);
            partition.assign_node(node.id(), selected);
        }
        
        // Phase 2: Minimize memory transfers
        partition = self.optimize_memory_transfers(partition);
        
        // Phase 3: Load balancing
        partition = self.balance_load(partition);
        
        partition
    }
    
    fn find_capable_providers(&self, node: &GraphNode) -> Vec<ProviderId> {
        self.providers
            .iter()
            .filter(|(_, provider)| {
                provider.can_handle(&[node.operator_spec()]).get(0).copied().unwrap_or(false)
            })
            .map(|(id, _)| *id)
            .collect()
    }
    
    fn select_optimal_provider(&self, candidates: &[ProviderId], node: &GraphNode) -> ProviderId {
        candidates
            .iter()
            .min_by_key(|&&provider_id| {
                self.cost_estimator.estimate_cost(provider_id, node)
            })
            .copied()
            .unwrap_or(ProviderId::CPU) // Fallback to CPU
    }
}

pub struct GraphPartition {
    assignments: HashMap<NodeId, ProviderId>,
    subgraphs: HashMap<ProviderId, SubGraph>,
    memory_transfers: Vec<MemoryTransfer>,
}
```

## Memory Management

### Memory Transfer Optimization

When operations are assigned to different providers, the framework minimizes data transfers:

```rust
pub struct MemoryTransferOptimizer {
    transfer_costs: HashMap<(MemoryType, MemoryType), f64>,
}

impl MemoryTransferOptimizer {
    pub fn optimize_transfers(&self, partition: &mut GraphPartition) {
        // Identify necessary memory transfers
        let transfers = self.identify_transfers(partition);
        
        // Minimize transfer overhead by:
        // 1. Batching transfers
        // 2. Using asynchronous copies
        // 3. Leveraging shared memory when possible
        let optimized_transfers = self.optimize_transfer_schedule(transfers);
        
        partition.memory_transfers = optimized_transfers;
    }
    
    fn identify_transfers(&self, partition: &GraphPartition) -> Vec<MemoryTransfer> {
        let mut transfers = Vec::new();
        
        for (node_id, provider_id) in &partition.assignments {
            for input_edge in partition.get_input_edges(*node_id) {
                let src_provider = partition.assignments[&input_edge.src_node];
                if src_provider != *provider_id {
                    transfers.push(MemoryTransfer {
                        from_provider: src_provider,
                        to_provider: *provider_id,
                        tensor_id: input_edge.tensor_id,
                        size: input_edge.tensor_size,
                    });
                }
            }
        }
        
        transfers
    }
}
```

## Performance Optimization

### Kernel Fusion

The framework automatically fuses compatible operations to reduce memory bandwidth:

```rust
pub struct KernelFuser {
    fusion_patterns: Vec<FusionPattern>,
}

impl KernelFuser {
    pub fn fuse_kernels(&self, subgraph: &mut SubGraph) -> Result<()> {
        for pattern in &self.fusion_patterns {
            while let Some(match_result) = pattern.find_match(subgraph) {
                self.apply_fusion(subgraph, match_result)?;
            }
        }
        Ok(())
    }
}

pub struct FusionPattern {
    pub name: String,
    pub pattern: GraphPattern,
    pub fused_kernel: Box<dyn FusedKernel>,
}

// Example: Conv2D + BatchNorm + ReLU fusion
pub struct ConvBnReluFusion {
    conv_kernel: ConvKernel,
    bn_params: BatchNormParams,
}

impl FusedKernel for ConvBnReluFusion {
    fn execute(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Single-pass execution combining all three operations
        let conv_out = self.conv_kernel.execute_raw(inputs)?;
        let bn_out = self.apply_batchnorm_inline(&conv_out)?;
        let relu_out = self.apply_relu_inline(&bn_out)?;
        Ok(vec![relu_out])
    }
}
```

## Implementation Details

### Provider Configuration

```rust
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub thread_count: Option<usize>,
    pub memory_limit: Option<usize>,
    pub optimization_level: OptimizationLevel,
    pub custom_options: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Custom(Vec<OptimizationFlag>),
}

// Example configuration for different scenarios
impl ProviderConfig {
    pub fn edge_optimized() -> Self {
        Self {
            thread_count: Some(4),
            memory_limit: Some(1_000_000_000), // 1GB
            optimization_level: OptimizationLevel::Aggressive,
            custom_options: hashmap! {
                "prefer_memory_over_compute".to_string() => "true".to_string(),
                "enable_quantization".to_string() => "true".to_string(),
            },
        }
    }
    
    pub fn cloud_optimized() -> Self {
        Self {
            thread_count: None, // Use all available cores
            memory_limit: None, // No memory limit
            optimization_level: OptimizationLevel::Basic,
            custom_options: hashmap! {
                "enable_parallel_execution".to_string() => "true".to_string(),
                "batch_size_optimization".to_string() => "true".to_string(),
            },
        }
    }
}
```

### Error Handling and Fallbacks

```rust
pub struct ProviderExecutionContext {
    primary_provider: ProviderId,
    fallback_providers: Vec<ProviderId>,
    retry_count: usize,
}

impl ProviderExecutionContext {
    pub async fn execute_with_fallback(
        &self,
        kernel: &dyn CompiledKernel,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let mut last_error = None;
        
        // Try primary provider first
        match kernel.execute(inputs).await {
            Ok(outputs) => return Ok(outputs),
            Err(e) => {
                last_error = Some(e);
                log::warn!("Primary provider failed, trying fallbacks");
            }
        }
        
        // Try fallback providers
        for &fallback_id in &self.fallback_providers {
            if let Ok(fallback_kernel) = self.recompile_for_provider(kernel, fallback_id) {
                match fallback_kernel.execute(inputs).await {
                    Ok(outputs) => {
                        log::info!("Fallback provider {} succeeded", fallback_id);
                        return Ok(outputs);
                    }
                    Err(e) => {
                        last_error = Some(e);
                        continue;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow!("All providers failed")))
    }
}
```

## Testing Strategy

### Provider Validation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_provider_capability_detection() {
        let cpu_provider = CPUExecutionProvider::new().unwrap();
        let capability = cpu_provider.get_capability();
        
        assert!(capability.supported_ops.contains(&OperatorType::MatMul));
        assert!(capability.data_types.contains(&DataType::F32));
    }
    
    #[tokio::test]
    async fn test_cross_provider_execution() {
        let registry = setup_test_registry();
        let graph = create_test_graph();
        
        let partition = registry.partition_graph(&graph);
        let results = execute_partitioned_graph(&partition).await.unwrap();
        
        assert_results_are_correct(&results);
    }
    
    #[test]
    fn test_memory_transfer_optimization() {
        let optimizer = MemoryTransferOptimizer::new();
        let mut partition = create_test_partition();
        
        optimizer.optimize_transfers(&mut partition);
        
        // Verify transfers are minimized
        assert!(partition.memory_transfers.len() <= expected_min_transfers());
    }
}
```

This execution provider framework provides the foundation for high-performance, hardware-agnostic ML inference while maintaining the flexibility to leverage specialized accelerators and optimization techniques.