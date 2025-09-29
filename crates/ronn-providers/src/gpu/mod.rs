//! GPU execution provider module using Candle backend.
//!
//! This module provides GPU-accelerated execution using the Candle library
//! for CUDA and Metal backends.

pub mod allocator;
pub mod provider;

pub use allocator::GpuMemoryAllocator;
pub use provider::{create_gpu_provider, create_gpu_provider_with_config, GpuExecutionProvider};
