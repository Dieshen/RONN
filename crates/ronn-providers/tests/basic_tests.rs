//! Basic tests for RONN execution provider framework.
//!
//! These tests validate core functionality without complex dependencies.

use anyhow::Result;
use ronn_core::tensor::Tensor;
use ronn_core::{DataType, TensorLayout};
use ronn_core::{MemoryType, TensorAllocator};
use ronn_providers::{
    detect_simd_capabilities, AlignedMemoryAllocator, PoolConfig, PooledMemoryAllocator,
    SystemMemoryAllocator,
};

/// Test SIMD capability detection.
#[test]
fn test_simd_detection() -> Result<()> {
    let capabilities = detect_simd_capabilities();

    println!("🔍 Detected SIMD capabilities:");
    println!("  SSE2: {}", capabilities.sse2);
    println!("  SSE4.1: {}", capabilities.sse41);
    println!("  AVX: {}", capabilities.avx);
    println!("  AVX2: {}", capabilities.avx2);
    println!("  AVX-512F: {}", capabilities.avx512f);
    println!("  FMA: {}", capabilities.fma);

    // On x86_64, SSE2 should be available
    #[cfg(target_arch = "x86_64")]
    assert!(capabilities.sse2, "SSE2 should be available on x86_64");

    println!("✅ SIMD detection working correctly");
    Ok(())
}

/// Test basic tensor creation and properties.
#[test]
fn test_tensor_creation() -> Result<()> {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];

    let tensor = Tensor::from_data(
        data.clone(),
        shape.clone(),
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.dtype(), DataType::F32);
    assert_eq!(tensor.layout(), TensorLayout::RowMajor);

    println!(
        "✅ Created tensor with shape {:?} and {} elements",
        tensor.shape(),
        data.len()
    );
    Ok(())
}

/// Test system memory allocator.
#[test]
fn test_system_allocator() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();

    // Test allocation
    let buffer = allocator.allocate(&[100], DataType::F32)?;
    assert_eq!(buffer.size, 400); // 100 * 4 bytes for F32
    assert_eq!(buffer.memory_type, MemoryType::SystemRAM);
    assert!(buffer.alignment >= std::mem::align_of::<f32>());

    let memory_info_before = allocator.get_memory_info();
    println!(
        "💾 Memory allocated: {} bytes",
        memory_info_before.allocated_bytes
    );

    // Test deallocation
    allocator.deallocate(buffer)?;

    let memory_info_after = allocator.get_memory_info();
    assert!(memory_info_after.allocated_bytes <= memory_info_before.allocated_bytes);

    println!("✅ System allocator working correctly");
    Ok(())
}

/// Test aligned memory allocator.
#[test]
fn test_aligned_allocator() -> Result<()> {
    let allocator = AlignedMemoryAllocator::new();

    // Test allocation with alignment requirements
    let buffer = allocator.allocate(&[64], DataType::F32)?;
    assert_eq!(buffer.size, 256); // 64 * 4 bytes for F32
    assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

    // Should be aligned to at least SIMD requirements
    assert!(buffer.alignment >= 16, "Should have SIMD alignment");

    println!(
        "🎯 Aligned buffer: {} bytes with {}-byte alignment",
        buffer.size, buffer.alignment
    );

    allocator.deallocate(buffer)?;

    println!("✅ Aligned allocator working correctly");
    Ok(())
}

/// Test pooled memory allocator.
#[test]
fn test_pooled_allocator() -> Result<()> {
    let config = PoolConfig {
        max_buffers_per_bucket: 4,
        max_pool_size: 1024 * 1024, // 1MB
        bucket_granularity: 64,
    };

    let allocator = PooledMemoryAllocator::new(config);

    // Allocate and deallocate multiple buffers to test pooling
    let mut buffers = Vec::new();

    for i in 0..5 {
        let buffer = allocator.allocate(&[50 + i * 10], DataType::F32)?;
        buffers.push(buffer);
        println!("🪣 Allocated buffer {}: {} bytes", i, buffers[i].size);
    }

    let hit_rate_before = allocator.get_hit_rate();

    // Deallocate all buffers (should go to pool)
    for (i, buffer) in buffers.into_iter().enumerate() {
        allocator.deallocate(buffer)?;
        println!("♻️  Deallocated buffer {}", i);
    }

    // Allocate similar sizes again (should hit pool)
    let buffer1 = allocator.allocate(&[50], DataType::F32)?;
    let buffer2 = allocator.allocate(&[60], DataType::F32)?;

    let hit_rate_after = allocator.get_hit_rate();

    println!(
        "📈 Pool hit rate: {:.2}% -> {:.2}%",
        hit_rate_before * 100.0,
        hit_rate_after * 100.0
    );

    allocator.deallocate(buffer1)?;
    allocator.deallocate(buffer2)?;

    println!("✅ Pooled allocator working correctly");
    Ok(())
}

/// Test multiple tensor operations.
#[test]
fn test_tensor_operations() -> Result<()> {
    // Create test matrices
    let matrix_a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let matrix_b = Tensor::from_data(
        vec![0.5, 1.0, 1.5, 2.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    println!("🧮 Created matrices:");
    println!("  Matrix A: shape {:?}", matrix_a.shape());
    println!("  Matrix B: shape {:?}", matrix_b.shape());

    // Test basic properties
    assert_eq!(matrix_a.shape(), vec![2, 2]);
    assert_eq!(matrix_b.shape(), vec![2, 2]);
    assert_eq!(matrix_a.dtype(), DataType::F32);
    assert_eq!(matrix_b.dtype(), DataType::F32);

    // Test zeros creation
    let zeros = Tensor::zeros(vec![3, 3], DataType::F32, TensorLayout::RowMajor)?;
    assert_eq!(zeros.shape(), vec![3, 3]);
    println!("  Zeros matrix: shape {:?}", zeros.shape());

    // Test ones creation
    let ones = Tensor::ones(vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;
    assert_eq!(ones.shape(), vec![2, 3]);
    println!("  Ones matrix: shape {:?}", ones.shape());

    println!("✅ Tensor operations working correctly");
    Ok(())
}

/// Performance test for memory allocation speed.
#[test]
fn test_allocation_performance() -> Result<()> {
    use std::time::Instant;

    let allocator = PooledMemoryAllocator::new(PoolConfig::default());
    let num_allocations = 1000;

    let start = Instant::now();
    let mut buffers = Vec::new();

    // Allocate many buffers
    for i in 0..num_allocations {
        let size = 100 + (i % 10) * 10; // Vary sizes slightly
        let buffer = allocator.allocate(&[size], DataType::F32)?;
        buffers.push(buffer);
    }

    let alloc_time = start.elapsed();

    let start = Instant::now();

    // Deallocate all buffers
    for buffer in buffers {
        allocator.deallocate(buffer)?;
    }

    let dealloc_time = start.elapsed();

    println!("⚡ Performance results:");
    println!(
        "  {} allocations: {:?} ({:.2} μs/alloc)",
        num_allocations,
        alloc_time,
        alloc_time.as_micros() as f64 / num_allocations as f64
    );
    println!(
        "  {} deallocations: {:?} ({:.2} μs/dealloc)",
        num_allocations,
        dealloc_time,
        dealloc_time.as_micros() as f64 / num_allocations as f64
    );

    // Should be reasonably fast
    assert!(
        alloc_time.as_millis() < 100,
        "Allocations should be under 100ms"
    );
    assert!(
        dealloc_time.as_millis() < 50,
        "Deallocations should be under 50ms"
    );

    println!("✅ Allocation performance is acceptable");
    Ok(())
}

/// Test memory info tracking.
#[test]
fn test_memory_tracking() -> Result<()> {
    let allocator = SystemMemoryAllocator::new();

    let initial_info = allocator.get_memory_info();
    assert_eq!(initial_info.allocated_bytes, 0);

    // Allocate multiple buffers
    let buffer1 = allocator.allocate(&[100], DataType::F32)?;
    let buffer2 = allocator.allocate(&[200], DataType::F32)?;

    let allocated_info = allocator.get_memory_info();
    let expected_size = 100 * 4 + 200 * 4; // F32 is 4 bytes
    assert_eq!(allocated_info.allocated_bytes, expected_size);
    assert!(allocated_info.peak_bytes >= expected_size);

    println!("📊 Memory tracking:");
    println!("  Initial: {} bytes", initial_info.allocated_bytes);
    println!(
        "  After allocation: {} bytes",
        allocated_info.allocated_bytes
    );
    println!("  Peak: {} bytes", allocated_info.peak_bytes);

    // Deallocate one buffer
    allocator.deallocate(buffer1)?;
    let partial_info = allocator.get_memory_info();
    assert_eq!(partial_info.allocated_bytes, 200 * 4);

    // Deallocate remaining buffer
    allocator.deallocate(buffer2)?;
    let final_info = allocator.get_memory_info();
    assert_eq!(final_info.allocated_bytes, 0);

    println!("  After cleanup: {} bytes", final_info.allocated_bytes);
    println!("✅ Memory tracking working correctly");
    Ok(())
}

/// Integration test combining multiple components.
#[test]
fn test_integration_workflow() -> Result<()> {
    println!("🚀 Running integration workflow test...");

    // 1. Test SIMD detection
    let simd_caps = detect_simd_capabilities();
    println!(
        "   ✓ SIMD capabilities detected: AVX2={}, FMA={}",
        simd_caps.avx2, simd_caps.fma
    );

    // 2. Create tensors
    let tensor_a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let tensor_b = Tensor::ones(vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;
    println!(
        "   ✓ Tensors created: A={:?}, B={:?}",
        tensor_a.shape(),
        tensor_b.shape()
    );

    // 3. Test different allocators
    let sys_alloc = SystemMemoryAllocator::new();
    let aligned_alloc = AlignedMemoryAllocator::new();
    let pooled_alloc = PooledMemoryAllocator::new(PoolConfig::default());

    // Test each allocator
    for (name, allocator) in [
        ("System", &sys_alloc as &dyn TensorAllocator),
        ("Aligned", &aligned_alloc as &dyn TensorAllocator),
        ("Pooled", &pooled_alloc as &dyn TensorAllocator),
    ] {
        let buffer = allocator.allocate(&[50], DataType::F32)?;
        let info = allocator.get_memory_info();
        println!(
            "   ✓ {} allocator: {} bytes allocated",
            name, info.allocated_bytes
        );
        allocator.deallocate(buffer)?;
    }

    println!("✅ Integration workflow test passed!");
    Ok(())
}
