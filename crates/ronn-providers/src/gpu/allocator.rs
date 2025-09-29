//! GPU memory allocator using Candle backend.
//!
//! This module provides GPU memory allocation and management through
//! the Candle library's device abstraction.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor as CandleTensor};
use ronn_core::{DataType, MemoryInfo, MemoryType, TensorAllocator, TensorBuffer};
use tracing::{debug, warn};

/// GPU memory allocator using Candle's device abstraction.
#[derive(Debug)]
pub struct GpuMemoryAllocator {
    /// GPU device for allocation.
    device: Device,
    /// Memory usage statistics.
    stats: Arc<Mutex<GpuMemoryStats>>,
    /// Memory pool for buffer reuse.
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
}

#[derive(Debug, Default)]
struct GpuMemoryStats {
    allocated_bytes: usize,
    peak_bytes: usize,
    allocation_count: usize,
    deallocation_count: usize,
}

#[derive(Debug, Default)]
struct GpuMemoryPool {
    // Simple pool implementation - in practice would be more sophisticated
    cached_buffers: Vec<(usize, TensorBuffer)>, // (size, buffer) pairs
    max_pool_size: usize,
    current_pool_size: usize,
}

impl GpuMemoryAllocator {
    /// Create a new GPU memory allocator for the specified device.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            stats: Arc::new(Mutex::new(GpuMemoryStats::default())),
            memory_pool: Arc::new(Mutex::new(GpuMemoryPool {
                cached_buffers: Vec::new(),
                max_pool_size: 256 * 1024 * 1024, // 256MB pool
                current_pool_size: 0,
            })),
        }
    }

    /// Create a CUDA GPU allocator.
    #[cfg(feature = "gpu")]
    pub fn new_cuda(device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)
            .map_err(|e| anyhow!("Failed to create CUDA device {}: {}", device_id, e))?;
        Ok(Self::new(device))
    }

    /// Create a Metal GPU allocator (macOS).
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    pub fn new_metal() -> Result<Self> {
        let device =
            Device::new_metal(0).map_err(|e| anyhow!("Failed to create Metal device: {}", e))?;
        Ok(Self::new(device))
    }

    /// Get the underlying GPU device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if the device is CUDA.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Check if the device is Metal.
    pub fn is_metal(&self) -> bool {
        matches!(self.device, Device::Metal(_))
    }

    /// Get device information string.
    pub fn device_info(&self) -> String {
        match &self.device {
            Device::Cpu => "CPU".to_string(),
            Device::Cuda(_) => "CUDA".to_string(),
            Device::Metal(_) => "Metal".to_string(),
        }
    }

    /// Convert RONN DataType to Candle DType for GPU operations.
    fn dtype_to_candle(&self, dtype: DataType) -> candle_core::DType {
        match dtype {
            DataType::F32 => candle_core::DType::F32,
            DataType::F16 => candle_core::DType::F16,
            DataType::F64 => candle_core::DType::F64,
            DataType::U8 => candle_core::DType::U8,
            DataType::U32 => candle_core::DType::U32,
            // Fallback for unsupported types
            DataType::I8 | DataType::I32 | DataType::Bool => candle_core::DType::F32,
        }
    }

    /// Calculate element size for a data type.
    fn element_size(&self, dtype: DataType) -> usize {
        match dtype {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F16 => 2,
            DataType::F64 => 8,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
        }
    }

    /// Try to get a buffer from the memory pool.
    fn try_get_from_pool(&self, size: usize) -> Option<TensorBuffer> {
        let mut pool = self.memory_pool.lock().unwrap();

        // Find a buffer of appropriate size
        for i in 0..pool.cached_buffers.len() {
            if pool.cached_buffers[i].0 >= size && pool.cached_buffers[i].0 <= size * 2 {
                let (_, buffer) = pool.cached_buffers.swap_remove(i);
                pool.current_pool_size -= buffer.size;
                debug!("GPU pool hit: reusing buffer of size {} bytes", size);
                return Some(buffer);
            }
        }

        None
    }

    /// Return a buffer to the memory pool.
    fn return_to_pool(&self, buffer: TensorBuffer) -> bool {
        let mut pool = self.memory_pool.lock().unwrap();

        // Check if pool has space
        if pool.current_pool_size + buffer.size > pool.max_pool_size {
            return false;
        }

        let buffer_size = buffer.size;
        pool.cached_buffers.push((buffer_size, buffer));
        pool.current_pool_size += buffer_size;
        debug!("Returned GPU buffer of size {} bytes to pool", buffer_size);

        true
    }

    /// Allocate GPU memory using a dummy tensor approach.
    fn allocate_gpu_memory(&self, size: usize, dtype: DataType) -> Result<TensorBuffer> {
        let elements = size / self.element_size(dtype);
        let candle_dtype = self.dtype_to_candle(dtype);

        // Create a dummy tensor to allocate GPU memory
        let tensor = CandleTensor::zeros(&[elements], candle_dtype, &self.device)
            .map_err(|e| anyhow!("GPU memory allocation failed: {}", e))?;

        // In a real implementation, we would extract the raw pointer
        // For now, we'll simulate it with a null pointer and proper size
        let ptr = std::ptr::null_mut(); // This would be the actual GPU pointer

        Ok(TensorBuffer {
            ptr,
            size,
            alignment: 256, // GPU memory alignment is typically 256 bytes
            memory_type: MemoryType::DeviceMemory,
        })
    }

    /// Get detailed GPU memory statistics.
    pub fn get_gpu_stats(&self) -> GpuMemoryStats {
        let stats = self.stats.lock().unwrap();
        GpuMemoryStats {
            allocated_bytes: stats.allocated_bytes,
            peak_bytes: stats.peak_bytes,
            allocation_count: stats.allocation_count,
            deallocation_count: stats.deallocation_count,
        }
    }

    /// Clear the memory pool.
    pub fn clear_pool(&self) {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.cached_buffers.clear();
        pool.current_pool_size = 0;
        debug!("Cleared GPU memory pool");
    }

    /// Get memory pool statistics.
    pub fn get_pool_stats(&self) -> (usize, usize, usize) {
        let pool = self.memory_pool.lock().unwrap();
        (
            pool.cached_buffers.len(),
            pool.current_pool_size,
            pool.max_pool_size,
        )
    }
}

impl TensorAllocator for GpuMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = shape.iter().product::<usize>() * self.element_size(dtype);

        if size == 0 {
            return Err(anyhow!("Cannot allocate zero-sized tensor"));
        }

        // Try to get from pool first
        if let Some(buffer) = self.try_get_from_pool(size) {
            return Ok(buffer);
        }

        // Allocate new GPU memory
        let buffer = self.allocate_gpu_memory(size, dtype)?;

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocated_bytes += size;
            stats.peak_bytes = stats.peak_bytes.max(stats.allocated_bytes);
            stats.allocation_count += 1;
        }

        debug!(
            "Allocated {} bytes on GPU device: {}",
            size,
            self.device_info()
        );

        Ok(buffer)
    }

    fn deallocate(&self, buffer: TensorBuffer) -> Result<()> {
        if buffer.size == 0 {
            return Ok(());
        }

        let buffer_size = buffer.size;

        // Try to return to pool
        if self.return_to_pool(buffer) {
            return Ok(());
        }

        // Pool is full, deallocate immediately
        // In a real implementation, we would free the GPU memory here
        debug!(
            "Deallocated {} bytes from GPU device: {}",
            buffer_size,
            self.device_info()
        );

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocated_bytes = stats.allocated_bytes.saturating_sub(buffer_size);
            stats.deallocation_count += 1;
        }

        Ok(())
    }

    fn get_memory_info(&self) -> MemoryInfo {
        let stats = self.stats.lock().unwrap();

        // In a real implementation, we would query the GPU for total memory
        let total_bytes = match &self.device {
            Device::Cuda(_) => 8 * 1024 * 1024 * 1024,  // Assume 8GB
            Device::Metal(_) => 8 * 1024 * 1024 * 1024, // Assume 8GB
            _ => usize::MAX,
        };

        MemoryInfo {
            total_bytes,
            allocated_bytes: stats.allocated_bytes,
            peak_bytes: stats.peak_bytes,
        }
    }
}

/// Create a CUDA GPU allocator.
#[cfg(feature = "gpu")]
pub fn create_cuda_allocator(device_id: usize) -> Result<Arc<dyn TensorAllocator>> {
    Ok(Arc::new(GpuMemoryAllocator::new_cuda(device_id)?))
}

/// Create a Metal GPU allocator.
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub fn create_metal_allocator() -> Result<Arc<dyn TensorAllocator>> {
    Ok(Arc::new(GpuMemoryAllocator::new_metal()?))
}

/// Create a GPU allocator for the best available device.
#[cfg(feature = "gpu")]
pub fn create_gpu_allocator() -> Result<Arc<dyn TensorAllocator>> {
    // Try CUDA first
    if let Ok(allocator) = create_cuda_allocator(0) {
        return Ok(allocator);
    }

    // Try Metal on macOS
    #[cfg(target_os = "macos")]
    {
        if let Ok(allocator) = create_metal_allocator() {
            return Ok(allocator);
        }
    }

    Err(anyhow!("No GPU devices available"))
}

/// Fallback for when GPU is not available.
#[cfg(not(feature = "gpu"))]
pub fn create_gpu_allocator() -> Result<Arc<dyn TensorAllocator>> {
    Err(anyhow!("GPU support not compiled in"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests only run when GPU feature is enabled
    #[cfg(feature = "gpu")]
    mod gpu_tests {
        use super::*;

        #[test]
        fn test_gpu_allocator_creation() {
            // This test might fail if no GPU is available
            if let Ok(allocator) = GpuMemoryAllocator::new_cuda(0) {
                assert!(allocator.is_cuda());
                assert!(!allocator.is_metal());
                println!("Created CUDA allocator: {}", allocator.device_info());
            }
        }

        #[test]
        fn test_dtype_conversion() {
            let device = Device::Cpu; // Use CPU for testing
            let allocator = GpuMemoryAllocator::new(device);

            assert_eq!(
                allocator.dtype_to_candle(DataType::F32),
                candle_core::DType::F32
            );
            assert_eq!(
                allocator.dtype_to_candle(DataType::F16),
                candle_core::DType::F16
            );
            assert_eq!(
                allocator.dtype_to_candle(DataType::U8),
                candle_core::DType::U8
            );
        }

        #[test]
        fn test_element_sizes() {
            let device = Device::Cpu;
            let allocator = GpuMemoryAllocator::new(device);

            assert_eq!(allocator.element_size(DataType::F32), 4);
            assert_eq!(allocator.element_size(DataType::F16), 2);
            assert_eq!(allocator.element_size(DataType::F64), 8);
            assert_eq!(allocator.element_size(DataType::U8), 1);
        }
    }

    #[test]
    fn test_fallback_when_no_gpu() {
        // This should work regardless of GPU availability
        let device = Device::Cpu;
        let allocator = GpuMemoryAllocator::new(device);

        assert!(!allocator.is_cuda());
        assert!(!allocator.is_metal());
        assert_eq!(allocator.device_info(), "CPU");
    }

    #[test]
    fn test_memory_pool_operations() {
        let device = Device::Cpu;
        let allocator = GpuMemoryAllocator::new(device);

        // Test pool statistics
        let (count, size, max_size) = allocator.get_pool_stats();
        assert_eq!(count, 0);
        assert_eq!(size, 0);
        assert!(max_size > 0);

        // Test pool clearing
        allocator.clear_pool();

        let (count_after, size_after, _) = allocator.get_pool_stats();
        assert_eq!(count_after, 0);
        assert_eq!(size_after, 0);
    }

    #[test]
    fn test_gpu_memory_stats() {
        let device = Device::Cpu;
        let allocator = GpuMemoryAllocator::new(device);

        let stats = allocator.get_gpu_stats();
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.deallocation_count, 0);
    }
}
