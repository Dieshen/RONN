//! GPU execution provider module using Candle backend.
//!
//! This module provides GPU-accelerated execution using the Candle library
//! for CUDA and Metal backends.

pub mod allocator;
pub mod cuda_kernels;
pub mod memory_manager;
pub mod provider;
pub mod topology;

pub use allocator::GpuMemoryAllocator;
pub use cuda_kernels::{CudaKernelManager, CudaCompileOptions, CompiledCudaKernel, KernelLaunchConfig};
pub use memory_manager::{MultiGpuMemoryManager, MultiGpuMemoryConfig, SyncStrategy};
pub use provider::{create_gpu_provider, create_gpu_provider_with_config, GpuExecutionProvider};
pub use topology::{
    GpuTopologyManager, TopologyConfig, GpuTopology, PlacementPlan, Workload, WorkloadType,
    PlacementStrategy, LocalityAwarePlacement, BandwidthOptimizedPlacement, PowerEfficientPlacement
};
