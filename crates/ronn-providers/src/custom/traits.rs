//! Traits and interfaces for custom hardware providers.
//!
//! This module defines the core traits that custom hardware providers must
//! implement to integrate with the RONN execution framework.

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

use anyhow::Result;
use ronn_core::{CompiledKernel, DataType, KernelStats, MemoryUsage, SubGraph, Tensor};

/// Core trait for custom hardware execution providers.
pub trait CustomHardwareProvider: Send + Sync + Debug {
    /// Get the unique identifier for this hardware provider.
    fn provider_name(&self) -> &str;

    /// Get the hardware capabilities of this provider.
    fn get_hardware_capability(&self) -> HardwareCapability;

    /// Check if the hardware is available and functional.
    fn is_hardware_available(&self) -> bool;

    /// Initialize the hardware provider.
    fn initialize(&mut self) -> Result<()>;

    /// Compile a subgraph for execution on this hardware.
    fn compile_subgraph(&self, subgraph: &SubGraph) -> Result<Box<dyn CustomKernel>>;

    /// Get the device memory manager for this provider.
    fn get_device_memory(&self) -> &dyn DeviceMemory;

    /// Get performance statistics for this provider.
    fn get_performance_stats(&self) -> ProviderStats;

    /// Shutdown and cleanup resources.
    fn shutdown(&mut self) -> Result<()>;

    /// Get provider-specific configuration as any type.
    fn as_any(&self) -> &dyn Any;

    /// Get mutable provider-specific configuration as any type.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Hardware capability information.
#[derive(Debug, Clone)]
pub struct HardwareCapability {
    /// Hardware vendor (e.g., "Google", "Intel", "Qualcomm").
    pub vendor: String,
    /// Hardware model (e.g., "TPU v4", "Hexagon 888", "Movidius VPU").
    pub model: String,
    /// Architecture version.
    pub architecture_version: String,
    /// Supported data types.
    pub supported_data_types: Vec<DataType>,
    /// Maximum memory in bytes.
    pub max_memory_bytes: u64,
    /// Peak compute performance in TOPS (Tera Operations Per Second).
    pub peak_tops: f64,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Supported operation types.
    pub supported_operations: Vec<String>,
    /// Hardware-specific features.
    pub features: HashMap<String, String>,
    /// Power consumption characteristics.
    pub power_profile: PowerProfile,
}

/// Power consumption profile for the hardware.
#[derive(Debug, Clone)]
pub struct PowerProfile {
    /// Idle power consumption in watts.
    pub idle_power_watts: f64,
    /// Peak power consumption in watts.
    pub peak_power_watts: f64,
    /// Thermal design power (TDP) in watts.
    pub tdp_watts: f64,
    /// Power efficiency in TOPS/W.
    pub efficiency_tops_per_watt: f64,
}

/// Performance statistics for a custom hardware provider.
#[derive(Debug, Clone)]
pub struct ProviderStats {
    /// Total number of operations executed.
    pub total_operations: u64,
    /// Average execution time in microseconds.
    pub average_execution_time_us: f64,
    /// Current memory usage in bytes.
    pub memory_usage_bytes: u64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Hardware utilization percentage (0.0 to 100.0).
    pub hardware_utilization: f64,
    /// Current power consumption in watts.
    pub current_power_watts: f64,
    /// Total energy consumed in joules.
    pub total_energy_joules: f64,
}

/// Trait for compiled kernels on custom hardware.
pub trait CustomKernel: Send + Sync + Debug {
    /// Execute the kernel with the given inputs.
    fn execute(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;

    /// Get memory usage information for this kernel.
    fn get_memory_usage(&self) -> MemoryUsage;

    /// Get performance statistics for this kernel.
    fn get_performance_stats(&self) -> KernelStats;

    /// Get hardware-specific kernel information.
    fn get_kernel_info(&self) -> KernelInfo;

    /// Warm up the kernel (optional pre-compilation/caching).
    fn warmup(&self) -> Result<()> {
        Ok(()) // Default implementation
    }
}

/// Information about a compiled kernel.
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Kernel name or identifier.
    pub name: String,
    /// Operations included in this kernel.
    pub operations: Vec<String>,
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: u64,
    /// Expected execution time in microseconds.
    pub estimated_execution_time_us: f64,
    /// Hardware utilization level (0.0 to 1.0).
    pub hardware_utilization: f64,
    /// Kernel compilation time in milliseconds.
    pub compilation_time_ms: f64,
}

/// Trait for device memory management on custom hardware.
pub trait DeviceMemory: Send + Sync + Debug {
    /// Allocate memory on the device.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;

    /// Deallocate device memory.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Copy data from host to device.
    fn copy_to_device(&self, host_data: &[u8], device_buffer: &DeviceBuffer) -> Result<()>;

    /// Copy data from device to host.
    fn copy_from_device(&self, device_buffer: &DeviceBuffer, host_data: &mut [u8]) -> Result<()>;

    /// Get memory usage information.
    fn get_memory_info(&self) -> DeviceMemoryInfo;

    /// Synchronize device operations.
    fn synchronize(&self) -> Result<()>;

    /// Check if two buffers can be used together (e.g., same memory space).
    fn can_access(&self, buffer1: &DeviceBuffer, buffer2: &DeviceBuffer) -> bool;
}

/// Device buffer handle.
#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    /// Device-specific buffer handle.
    pub handle: u64,
    /// Buffer size in bytes.
    pub size: usize,
    /// Buffer alignment.
    pub alignment: usize,
    /// Device identifier.
    pub device_id: u32,
    /// Memory type (device-specific).
    pub memory_type: String,
}

/// Device memory information.
#[derive(Debug, Clone)]
pub struct DeviceMemoryInfo {
    /// Total device memory in bytes.
    pub total_bytes: u64,
    /// Available memory in bytes.
    pub available_bytes: u64,
    /// Currently allocated memory in bytes.
    pub allocated_bytes: u64,
    /// Memory bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Memory type description.
    pub memory_type: String,
}

/// Kernel compilation options for custom hardware.
#[derive(Debug, Clone)]
pub struct CompilationOptions {
    /// Optimization level (0-3).
    pub optimization_level: u8,
    /// Enable aggressive optimizations.
    pub aggressive_optimization: bool,
    /// Target precision (e.g., "fp32", "fp16", "int8").
    pub target_precision: String,
    /// Hardware-specific compiler flags.
    pub compiler_flags: Vec<String>,
    /// Custom defines for compilation.
    pub defines: HashMap<String, String>,
    /// Include paths for custom headers.
    pub include_paths: Vec<String>,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            aggressive_optimization: false,
            target_precision: "fp32".to_string(),
            compiler_flags: Vec::new(),
            defines: HashMap::new(),
            include_paths: Vec::new(),
        }
    }
}

/// Hardware discovery trait for detecting available custom hardware.
pub trait HardwareDiscovery: Send + Sync {
    /// Discover available hardware devices.
    fn discover_devices(&self) -> Result<Vec<HardwareDevice>>;

    /// Check if a specific device is available.
    fn is_device_available(&self, device_id: &str) -> bool;

    /// Get detailed information about a device.
    fn get_device_info(&self, device_id: &str) -> Option<HardwareDevice>;
}

/// Information about a discovered hardware device.
#[derive(Debug, Clone)]
pub struct HardwareDevice {
    /// Unique device identifier.
    pub device_id: String,
    /// Device name.
    pub name: String,
    /// Vendor name.
    pub vendor: String,
    /// Device type (e.g., "NPU", "TPU", "VPU").
    pub device_type: String,
    /// Driver version.
    pub driver_version: String,
    /// Firmware version.
    pub firmware_version: String,
    /// Device capabilities.
    pub capabilities: HardwareCapability,
    /// Current device status.
    pub status: DeviceStatus,
}

/// Status of a hardware device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Device is available and ready.
    Available,
    /// Device is busy with another task.
    Busy,
    /// Device has an error.
    Error(String),
    /// Device is not properly initialized.
    NotInitialized,
    /// Device is offline.
    Offline,
}

/// Profiling interface for custom hardware providers.
pub trait HardwareProfiler: Send + Sync {
    /// Start profiling a specific operation.
    fn start_profiling(&mut self, operation_name: &str) -> Result<ProfilingSession>;

    /// Stop profiling and get results.
    fn stop_profiling(&mut self, session: ProfilingSession) -> Result<ProfilingResults>;

    /// Get overall profiling summary.
    fn get_profiling_summary(&self) -> ProfilingSummary;

    /// Reset profiling data.
    fn reset_profiling(&mut self);
}

/// Profiling session handle.
#[derive(Debug)]
pub struct ProfilingSession {
    /// Session identifier.
    pub session_id: u64,
    /// Operation being profiled.
    pub operation_name: String,
    /// Start timestamp.
    pub start_time: std::time::Instant,
}

/// Results from a profiling session.
#[derive(Debug, Clone)]
pub struct ProfilingResults {
    /// Operation name.
    pub operation_name: String,
    /// Execution time in microseconds.
    pub execution_time_us: f64,
    /// Memory usage during execution.
    pub memory_usage_bytes: u64,
    /// Hardware utilization percentage.
    pub hardware_utilization: f64,
    /// Power consumption during execution.
    pub power_consumption_watts: f64,
    /// Energy consumed in millijoules.
    pub energy_consumed_mj: f64,
    /// Hardware-specific metrics.
    pub custom_metrics: HashMap<String, f64>,
}

/// Summary of all profiling data.
#[derive(Debug, Clone)]
pub struct ProfilingSummary {
    /// Total operations profiled.
    pub total_operations: u64,
    /// Total execution time across all operations.
    pub total_execution_time_us: f64,
    /// Average execution time per operation.
    pub average_execution_time_us: f64,
    /// Peak memory usage observed.
    pub peak_memory_bytes: u64,
    /// Average hardware utilization.
    pub average_utilization: f64,
    /// Total energy consumed.
    pub total_energy_joules: f64,
    /// Most expensive operations by time.
    pub top_operations_by_time: Vec<(String, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_capability_creation() {
        let capability = HardwareCapability {
            vendor: "TestVendor".to_string(),
            model: "TestAccelerator".to_string(),
            architecture_version: "1.0".to_string(),
            supported_data_types: vec![DataType::F32, DataType::F16],
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            peak_tops: 100.0,
            memory_bandwidth_gbps: 900.0,
            supported_operations: vec!["MatMul".to_string(), "Conv".to_string()],
            features: HashMap::new(),
            power_profile: PowerProfile {
                idle_power_watts: 5.0,
                peak_power_watts: 75.0,
                tdp_watts: 50.0,
                efficiency_tops_per_watt: 2.0,
            },
        };

        assert_eq!(capability.vendor, "TestVendor");
        assert_eq!(capability.peak_tops, 100.0);
        assert!(capability.supported_data_types.contains(&DataType::F32));
    }

    #[test]
    fn test_device_status() {
        let status = DeviceStatus::Available;
        assert_eq!(status, DeviceStatus::Available);

        let error_status = DeviceStatus::Error("Hardware fault".to_string());
        match error_status {
            DeviceStatus::Error(msg) => assert_eq!(msg, "Hardware fault"),
            _ => panic!("Expected error status"),
        }
    }

    #[test]
    fn test_compilation_options() {
        let options = CompilationOptions {
            optimization_level: 3,
            target_precision: "fp16".to_string(),
            ..Default::default()
        };

        assert_eq!(options.optimization_level, 3);
        assert_eq!(options.target_precision, "fp16");
        assert!(!options.aggressive_optimization);
    }

    #[test]
    fn test_device_buffer() {
        let buffer = DeviceBuffer {
            handle: 0x12345678,
            size: 1024,
            alignment: 256,
            device_id: 0,
            memory_type: "HBM".to_string(),
        };

        assert_eq!(buffer.handle, 0x12345678);
        assert_eq!(buffer.size, 1024);
        assert_eq!(buffer.memory_type, "HBM");
    }

    #[test]
    fn test_profiling_session() {
        let session = ProfilingSession {
            session_id: 1,
            operation_name: "test_op".to_string(),
            start_time: std::time::Instant::now(),
        };

        assert_eq!(session.session_id, 1);
        assert_eq!(session.operation_name, "test_op");
    }
}
