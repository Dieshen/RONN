//! Multi-GPU Integration Tests
//!
//! Comprehensive integration tests covering:
//! - Multi-GPU memory management and synchronization
//! - CUDA kernel compilation and execution
//! - Topology discovery and placement optimization
//! - Error handling and edge cases
//! - Performance regression detection

use ronn_core::{DataType, Tensor, TensorLayout, SubGraph, GraphNode, ProviderId};
use ronn_providers::{
    create_gpu_provider, create_gpu_provider_with_config, GpuExecutionProvider,
    MultiGpuMemoryManager, MultiGpuMemoryConfig, SyncStrategy, CudaKernelManager,
    CudaCompileOptions, GpuTopologyManager, TopologyConfig, Workload, WorkloadType,
    LocalityAwarePlacement, BandwidthOptimizedPlacement, PowerEfficientPlacement,
    PlacementStrategy, CommunicationPattern
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

/// Test multi-GPU memory management functionality
#[tokio::test]
async fn test_multi_gpu_memory_management() -> anyhow::Result<()> {
    // Skip if GPU not available
    if create_gpu_provider().is_err() {
        println!("GPU not available, skipping multi-GPU memory tests");
        return Ok(());
    }

    let config = MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 128 * 1024 * 1024, // 128MB per GPU
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: true,
    };

    let memory_manager = MultiGpuMemoryManager::new(config).await?;
    let memory_manager = Arc::new(memory_manager);

    // Test 1: Basic tensor allocation on specific device
    let test_data = vec![1.0f32; 1024];
    let tensor = Tensor::from_data(test_data, vec![1024], DataType::F32, TensorLayout::RowMajor)?;

    let device_tensor = memory_manager.allocate_on_device(tensor, 0).await?;
    assert_eq!(device_tensor.device_id, Some(0));

    // Test 2: Multi-device distribution
    let large_data = vec![2.0f32; 4096];
    let large_tensor = Tensor::from_data(large_data, vec![4096], DataType::F32, TensorLayout::RowMajor)?;

    let distributed_tensors = memory_manager.distribute_tensor(large_tensor, &[0]).await?;
    assert!(!distributed_tensors.is_empty());

    // Test 3: Memory usage tracking
    let memory_info = memory_manager.get_memory_info().await?;
    assert!(memory_info.allocated_bytes > 0);
    assert!(memory_info.available_bytes > 0);

    // Test 4: Synchronization
    memory_manager.synchronize_devices(&[0]).await?;

    Ok(())
}

/// Test peer-to-peer memory transfers
#[tokio::test]
async fn test_peer_to_peer_transfers() -> anyhow::Result<()> {
    let config = MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 128 * 1024 * 1024,
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: true,
    };

    let memory_manager = match MultiGpuMemoryManager::new(config).await {
        Ok(mm) => Arc::new(mm),
        Err(_) => {
            println!("Failed to create multi-GPU memory manager, skipping P2P tests");
            return Ok(());
        }
    };

    // Test P2P capability detection
    let can_p2p = memory_manager.can_access_peer(0, 1).await.unwrap_or(false);
    println!("P2P access between devices 0 and 1: {}", can_p2p);

    if !can_p2p {
        println!("P2P not available, skipping transfer tests");
        return Ok(());
    }

    // Test P2P transfer
    let test_data = vec![3.14f32; 2048];
    let tensor = Tensor::from_data(test_data, vec![2048], DataType::F32, TensorLayout::RowMajor)?;

    let device0_tensor = memory_manager.allocate_on_device(tensor, 0).await?;
    let device1_tensor = memory_manager.transfer_between_devices(&device0_tensor, 0, 1).await?;

    assert_eq!(device1_tensor.device_id, Some(1));

    Ok(())
}

/// Test CUDA kernel compilation and execution
#[tokio::test]
async fn test_cuda_kernel_execution() -> anyhow::Result<()> {
    let kernel_manager = match CudaKernelManager::new() {
        Ok(km) => km,
        Err(_) => {
            println!("CUDA not available, skipping kernel tests");
            return Ok(());
        }
    };

    // Test 1: Basic kernel compilation
    let kernel_source = r#"
        extern "C" __global__ void vector_scale(float* data, float scale, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] *= scale;
            }
        }
    "#;

    let compile_options = CudaCompileOptions {
        optimization_level: 2,
        debug_info: false,
        fast_math: true,
        architecture: "sm_75".to_string(),
        include_paths: vec![],
        define_macros: HashMap::new(),
    };

    let compiled_kernel = kernel_manager.compile_kernel(
        kernel_source,
        "vector_scale",
        &compile_options
    )?;

    // Test 2: Kernel execution
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let scale = 2.0f32;
    let n = data.len();

    let kernel_args = vec![
        data.as_ptr() as *const u8,
        &scale as *const f32 as *const u8,
        &n as *const usize as *const u8,
    ];

    compiled_kernel.launch(&kernel_args, 1, 4)?;

    // Test 3: Kernel performance statistics
    let stats = compiled_kernel.get_execution_stats();
    assert!(stats.total_launches > 0);

    Ok(())
}

/// Test GPU topology discovery and management
#[tokio::test]
async fn test_gpu_topology_management() -> anyhow::Result<()> {
    let config = TopologyConfig {
        enable_numa_awareness: true,
        enable_bandwidth_profiling: false, // Disable for faster tests
        enable_power_monitoring: false,
        profiling_duration_ms: 50,
        cache_topology_info: true,
    };

    let topology_manager = match GpuTopologyManager::new(config) {
        Ok(tm) => tm,
        Err(_) => {
            println!("Failed to create topology manager, skipping topology tests");
            return Ok(());
        }
    };

    // Test 1: Topology discovery
    topology_manager.discover_topology().await?;

    let topology = topology_manager.get_topology().await?;
    assert!(!topology.devices.is_empty());

    // Test 2: Device capabilities
    for (device_id, device_info) in &topology.devices {
        println!("Device {}: {} with {}GB memory",
                 device_id, device_info.name, device_info.memory_total / (1024 * 1024 * 1024));
        assert!(device_info.memory_total > 0);
        assert!(!device_info.name.is_empty());
    }

    // Test 3: Interconnect detection
    println!("Detected {} interconnect links", topology.links.len());
    for ((from, to), link) in &topology.links {
        println!("Link {}->{}: {:?} ({}GB/s)", from, to, link.link_type, link.bandwidth_gbps);
    }

    Ok(())
}

/// Test placement strategies
#[tokio::test]
async fn test_placement_strategies() -> anyhow::Result<()> {
    let config = TopologyConfig {
        enable_numa_awareness: true,
        enable_bandwidth_profiling: false,
        enable_power_monitoring: false,
        profiling_duration_ms: 50,
        cache_topology_info: true,
    };

    let topology_manager = match GpuTopologyManager::new(config) {
        Ok(tm) => tm,
        Err(_) => {
            println!("Failed to create topology manager, skipping placement tests");
            return Ok(());
        }
    };

    topology_manager.discover_topology().await?;
    let topology = topology_manager.get_topology().await?;

    let strategies: Vec<Box<dyn PlacementStrategy + Send + Sync>> = vec![
        Box::new(LocalityAwarePlacement::new()),
        Box::new(BandwidthOptimizedPlacement::new()),
        Box::new(PowerEfficientPlacement::new()),
    ];

    // Test different workload types
    let workloads = vec![
        ("compute_intensive", Workload {
            id: "test_compute".to_string(),
            workload_type: WorkloadType::ComputeIntensive,
            estimated_compute_ops: 1_000_000,
            estimated_memory_usage: 256 * 1024 * 1024,
            communication_pattern: CommunicationPattern::AllToAll,
            priority: 1.0,
        }),
        ("memory_bound", Workload {
            id: "test_memory".to_string(),
            workload_type: WorkloadType::MemoryBound,
            estimated_compute_ops: 10_000,
            estimated_memory_usage: 1024 * 1024 * 1024,
            communication_pattern: CommunicationPattern::Broadcast,
            priority: 1.0,
        }),
        ("communication_heavy", Workload {
            id: "test_comm".to_string(),
            workload_type: WorkloadType::CommunicationHeavy,
            estimated_compute_ops: 100_000,
            estimated_memory_usage: 128 * 1024 * 1024,
            communication_pattern: CommunicationPattern::Ring,
            priority: 1.0,
        }),
    ];

    for (workload_name, workload) in workloads {
        for (i, strategy) in strategies.iter().enumerate() {
            let plan = strategy.create_placement_plan(&workload, &topology)?;

            println!("Strategy {} for {}: {} devices selected",
                     i, workload_name, plan.device_assignments.len());

            assert!(!plan.device_assignments.is_empty());
            assert!(plan.estimated_performance_score >= 0.0);
            assert!(plan.estimated_power_consumption >= 0.0);

            // Validate device assignments are within available devices
            for assignment in &plan.device_assignments {
                assert!(topology.devices.contains_key(&assignment.device_id));
                assert!(assignment.memory_allocation > 0);
                assert!(assignment.compute_allocation > 0.0 && assignment.compute_allocation <= 1.0);
            }
        }
    }

    Ok(())
}

/// Test end-to-end multi-GPU execution
#[tokio::test]
async fn test_end_to_end_execution() -> anyhow::Result<()> {
    let provider = match create_gpu_provider() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping end-to-end tests");
            return Ok(());
        }
    };

    // Create a test computation graph
    let subgraph = SubGraph {
        nodes: vec![
            GraphNode {
                id: 0,
                op_type: "MatMul".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["A".to_string(), "B".to_string()],
                outputs: vec!["C".to_string()],
                name: Some("matmul_test".to_string()),
            },
            GraphNode {
                id: 1,
                op_type: "Add".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["C".to_string(), "bias".to_string()],
                outputs: vec!["result".to_string()],
                name: Some("add_bias".to_string()),
            }
        ],
        edges: vec![],
        inputs: vec!["A".to_string(), "B".to_string(), "bias".to_string()],
        outputs: vec!["result".to_string()],
    };

    // Create test inputs
    let matrix_a = Tensor::ones(vec![64, 32], DataType::F32, TensorLayout::RowMajor)?;
    let matrix_b = Tensor::ones(vec![32, 48], DataType::F32, TensorLayout::RowMajor)?;
    let bias = Tensor::zeros(vec![64, 48], DataType::F32, TensorLayout::RowMajor)?;
    let inputs = vec![matrix_a, matrix_b, bias];

    // Test 1: Single device execution
    let start_time = std::time::Instant::now();
    let result = provider.execute_subgraph(&subgraph, &inputs).await?;
    let single_gpu_time = start_time.elapsed();

    assert!(!result.is_empty());
    println!("Single GPU execution time: {:?}", single_gpu_time);

    // Test 2: Multi-device execution with optimization
    if provider.get_topology().is_some() {
        let workload = Workload {
            id: "end_to_end_test".to_string(),
            workload_type: WorkloadType::ComputeIntensive,
            estimated_compute_ops: 64 * 32 * 48,
            estimated_memory_usage: (64 * 32 + 32 * 48 + 64 * 48) * std::mem::size_of::<f32>() as u64,
            communication_pattern: CommunicationPattern::PipelineParallel,
            priority: 1.0,
        };

        let optimal_devices = provider.auto_select_devices(&workload).await?;

        let start_time = std::time::Instant::now();
        let result = provider.execute_subgraph_on_devices(&subgraph, &inputs, &optimal_devices).await?;
        let multi_gpu_time = start_time.elapsed();

        assert!(!result.is_empty());
        println!("Multi-GPU execution time: {:?} on devices {:?}", multi_gpu_time, optimal_devices);
    }

    Ok(())
}

/// Test error handling and edge cases
#[tokio::test]
async fn test_error_handling() -> anyhow::Result<()> {
    // Test 1: Invalid device ID
    let config = MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 128 * 1024 * 1024,
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: true,
    };

    let memory_manager = MultiGpuMemoryManager::new(config).await.map(Arc::new);

    if let Ok(mm) = memory_manager {
        let test_data = vec![1.0f32; 100];
        let tensor = Tensor::from_data(test_data, vec![100], DataType::F32, TensorLayout::RowMajor)?;

        // Should fail with invalid device ID
        let result = mm.allocate_on_device(tensor, 999).await;
        assert!(result.is_err());
    }

    // Test 2: Invalid CUDA kernel compilation
    if let Ok(kernel_manager) = CudaKernelManager::new() {
        let invalid_kernel = "invalid C++ syntax !!!";

        let result = kernel_manager.compile_kernel(
            invalid_kernel,
            "invalid_kernel",
            &Default::default()
        );
        assert!(result.is_err());
    }

    // Test 3: Out of memory scenarios
    if let Ok(mm) = MultiGpuMemoryManager::new(MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 1024, // Very small pool
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: false,
    }).await {
        let huge_data = vec![0.0f32; 1024 * 1024 * 1024]; // 4GB
        let huge_tensor = Tensor::from_data(huge_data, vec![1024 * 1024 * 1024], DataType::F32, TensorLayout::RowMajor);

        if let Ok(tensor) = huge_tensor {
            let result = mm.allocate_on_device(tensor, 0).await;
            // Should fail due to insufficient memory
            assert!(result.is_err());
        }
    }

    Ok(())
}

/// Test concurrent multi-GPU operations
#[tokio::test]
async fn test_concurrent_operations() -> anyhow::Result<()> {
    let config = MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 256 * 1024 * 1024,
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: true,
    };

    let memory_manager = match MultiGpuMemoryManager::new(config).await {
        Ok(mm) => Arc::new(mm),
        Err(_) => {
            println!("Failed to create memory manager, skipping concurrent tests");
            return Ok(());
        }
    };

    // Launch multiple concurrent operations
    let tasks = (0..8).map(|i| {
        let mm = memory_manager.clone();
        tokio::spawn(async move {
            let data = vec![i as f32; 1024];
            let tensor = Tensor::from_data(data, vec![1024], DataType::F32, TensorLayout::RowMajor)?;

            let device_tensor = mm.allocate_on_device(tensor, 0).await?;

            // Simulate some work
            tokio::time::sleep(Duration::from_millis(10)).await;

            mm.deallocate(device_tensor).await?;

            Ok::<_, anyhow::Error>(i)
        })
    }).collect::<Vec<_>>();

    // Wait for all tasks to complete with timeout
    let results = timeout(Duration::from_secs(30), futures::future::try_join_all(tasks)).await;
    assert!(results.is_ok());

    let task_results = results.unwrap().unwrap();
    assert_eq!(task_results.len(), 8);

    Ok(())
}

/// Test performance regression detection
#[tokio::test]
async fn test_performance_regression() -> anyhow::Result<()> {
    let provider = match create_gpu_provider() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping performance tests");
            return Ok(());
        }
    };

    // Simple computation for baseline measurement
    let matrix_size = 256;
    let matrix_a = Tensor::ones(vec![matrix_size, matrix_size], DataType::F32, TensorLayout::RowMajor)?;
    let matrix_b = Tensor::ones(vec![matrix_size, matrix_size], DataType::F32, TensorLayout::RowMajor)?;
    let inputs = vec![matrix_a, matrix_b];

    let subgraph = SubGraph {
        nodes: vec![GraphNode {
            id: 0,
            op_type: "MatMul".to_string(),
            attributes: HashMap::new(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            name: Some("perf_test".to_string()),
        }],
        edges: vec![],
        inputs: vec!["A".to_string(), "B".to_string()],
        outputs: vec!["C".to_string()],
    };

    // Perform multiple runs to get stable measurements
    let mut execution_times = Vec::new();

    for _ in 0..10 {
        let start = std::time::Instant::now();
        let _result = provider.execute_subgraph(&subgraph, &inputs).await?;
        execution_times.push(start.elapsed());
    }

    let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
    let min_time = execution_times.iter().min().unwrap();
    let max_time = execution_times.iter().max().unwrap();

    println!("Performance metrics for {}x{} matrix multiplication:", matrix_size, matrix_size);
    println!("  Average: {:?}", avg_time);
    println!("  Min: {:?}", min_time);
    println!("  Max: {:?}", max_time);

    // Performance regression check (adjust thresholds as needed)
    let expected_max_time = Duration::from_millis(100); // 100ms threshold
    assert!(avg_time < expected_max_time,
            "Performance regression detected: average execution time {:?} exceeds threshold {:?}",
            avg_time, expected_max_time);

    Ok(())
}

/// Test memory leak detection
#[tokio::test]
async fn test_memory_leak_detection() -> anyhow::Result<()> {
    let config = MultiGpuMemoryConfig {
        enable_peer_to_peer: true,
        enable_unified_memory: false,
        memory_pool_size: 128 * 1024 * 1024,
        sync_strategy: SyncStrategy::Async,
        enable_memory_prefetching: false, // Disable for accurate tracking
    };

    let memory_manager = match MultiGpuMemoryManager::new(config).await {
        Ok(mm) => Arc::new(mm),
        Err(_) => {
            println!("Failed to create memory manager, skipping memory leak tests");
            return Ok(());
        }
    };

    // Get initial memory usage
    let initial_memory = memory_manager.get_memory_info().await?;
    println!("Initial allocated memory: {} bytes", initial_memory.allocated_bytes);

    // Perform many allocations and deallocations
    for i in 0..100 {
        let data = vec![i as f32; 1024];
        let tensor = Tensor::from_data(data, vec![1024], DataType::F32, TensorLayout::RowMajor)?;

        let device_tensor = memory_manager.allocate_on_device(tensor, 0).await?;
        memory_manager.deallocate(device_tensor).await?;
    }

    // Force garbage collection/cleanup
    memory_manager.synchronize_devices(&[0]).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check final memory usage
    let final_memory = memory_manager.get_memory_info().await?;
    println!("Final allocated memory: {} bytes", final_memory.allocated_bytes);

    // Memory should be back to initial levels (allowing some tolerance)
    let memory_increase = final_memory.allocated_bytes.saturating_sub(initial_memory.allocated_bytes);
    let max_acceptable_increase = 1024 * 1024; // 1MB tolerance

    assert!(memory_increase <= max_acceptable_increase,
            "Memory leak detected: allocated memory increased by {} bytes",
            memory_increase);

    Ok(())
}

/// Integration test for all components working together
#[tokio::test]
async fn test_full_system_integration() -> anyhow::Result<()> {
    // This test verifies that all multi-GPU components work together correctly

    // Step 1: Create GPU provider with topology
    let provider = match create_gpu_provider() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            println!("GPU not available, skipping full integration test");
            return Ok(());
        }
    };

    // Step 2: Verify topology detection works
    let topology = provider.get_topology();
    if topology.is_none() {
        println!("Topology not available, continuing with basic integration test");
    } else {
        println!("Topology detected with {} devices", topology.unwrap().devices.len());
    }

    // Step 3: Create complex workload
    let complex_workload = Workload {
        id: "integration_test".to_string(),
        workload_type: WorkloadType::ComputeIntensive,
        estimated_compute_ops: 1_000_000,
        estimated_memory_usage: 512 * 1024 * 1024, // 512MB
        communication_pattern: CommunicationPattern::AllToAll,
        priority: 1.0,
    };

    // Step 4: Test device selection
    let selected_devices = provider.auto_select_devices(&complex_workload).await.unwrap_or(vec![0]);
    println!("Selected devices: {:?}", selected_devices);

    // Step 5: Create and execute complex subgraph
    let complex_subgraph = SubGraph {
        nodes: vec![
            GraphNode {
                id: 0,
                op_type: "Conv2D".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["input".to_string(), "weights".to_string()],
                outputs: vec!["conv_out".to_string()],
                name: Some("conv_layer".to_string()),
            },
            GraphNode {
                id: 1,
                op_type: "BatchNorm".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["conv_out".to_string(), "bn_weights".to_string()],
                outputs: vec!["bn_out".to_string()],
                name: Some("batch_norm".to_string()),
            },
            GraphNode {
                id: 2,
                op_type: "ReLU".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["bn_out".to_string()],
                outputs: vec!["relu_out".to_string()],
                name: Some("activation".to_string()),
            },
            GraphNode {
                id: 3,
                op_type: "MaxPool".to_string(),
                attributes: HashMap::new(),
                inputs: vec!["relu_out".to_string()],
                outputs: vec!["final_out".to_string()],
                name: Some("pooling".to_string()),
            },
        ],
        edges: vec![],
        inputs: vec!["input".to_string(), "weights".to_string(), "bn_weights".to_string()],
        outputs: vec!["final_out".to_string()],
    };

    // Create realistic input tensors
    let input_tensor = Tensor::ones(vec![32, 64, 64, 3], DataType::F32, TensorLayout::RowMajor)?; // Batch of 32, 64x64 RGB images
    let weight_tensor = Tensor::ones(vec![16, 3, 3, 3], DataType::F32, TensorLayout::RowMajor)?; // 16 filters, 3x3 kernels
    let bn_weights = Tensor::ones(vec![16], DataType::F32, TensorLayout::RowMajor)?; // BatchNorm weights

    let inputs = vec![input_tensor, weight_tensor, bn_weights];

    // Step 6: Execute with timing
    let start_time = std::time::Instant::now();
    let execution_result = timeout(
        Duration::from_secs(60), // 60 second timeout
        provider.execute_subgraph_on_devices(&complex_subgraph, &inputs, &selected_devices)
    ).await;
    let execution_time = start_time.elapsed();

    // Step 7: Verify results
    assert!(execution_result.is_ok(), "Integration test timed out");
    let outputs = execution_result.unwrap()?;
    assert!(!outputs.is_empty(), "No outputs produced");

    println!("Full system integration test completed successfully");
    println!("Execution time: {:?}", execution_time);
    println!("Output tensors: {}", outputs.len());

    Ok(())
}