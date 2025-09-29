# Multi-GPU Testing Framework

This directory contains comprehensive tests and benchmarks for the RONN multi-GPU execution provider framework.

## Overview

The testing framework covers all aspects of multi-GPU functionality:

- **Memory Management**: Allocation, distribution, and synchronization across multiple GPUs
- **CUDA Kernels**: Compilation, execution, and performance optimization
- **Topology Detection**: GPU interconnect discovery and optimal placement strategies
- **Error Handling**: Edge cases, error recovery, and graceful degradation
- **Performance**: Benchmarking, regression detection, and memory leak checking

## Test Structure

### Unit Tests (`unit_tests.rs`)

Focused tests for individual components:
- `test_gpu_memory_allocator()` - Basic GPU memory allocation
- `test_cuda_compile_options()` - CUDA compilation configuration
- `test_cuda_kernel_manager()` - Kernel compilation and caching
- `test_multi_gpu_memory_config()` - Memory manager configuration
- `test_placement_strategies()` - All three placement algorithms
- `test_workload_characterization()` - Workload classification accuracy

### Integration Tests (`multi_gpu_integration_tests.rs`)

End-to-end tests covering complete workflows:
- `test_multi_gpu_memory_management()` - Full memory lifecycle
- `test_peer_to_peer_transfers()` - P2P memory transfers
- `test_cuda_kernel_execution()` - Kernel compilation and execution
- `test_gpu_topology_management()` - Topology discovery and analysis
- `test_placement_strategies()` - Placement algorithm integration
- `test_end_to_end_execution()` - Complete subgraph execution
- `test_error_handling()` - Error scenarios and recovery
- `test_concurrent_operations()` - Multi-threaded operations
- `test_performance_regression()` - Performance baseline checking
- `test_memory_leak_detection()` - Memory usage validation
- `test_full_system_integration()` - All components working together

### Performance Benchmarks (`../benches/multi_gpu_benchmarks.rs`)

Comprehensive performance measurements:
- `bench_memory_transfers()` - Memory allocation and transfer patterns
- `bench_cuda_kernels()` - Kernel execution performance
- `bench_topology_placement()` - Placement strategy efficiency
- `bench_end_to_end_execution()` - Complete execution pipelines
- `bench_load_balancing()` - Concurrent workload management

## Running Tests

### Quick Test Run

```bash
# Run all tests with GPU features
cargo test --features gpu

# Run specific test suite
cargo test --features gpu unit_tests
cargo test --features gpu multi_gpu_integration_tests
```

### Comprehensive Test Suite

Use the provided test runner scripts:

**Windows (PowerShell):**
```powershell
.\scripts\run_multi_gpu_tests.ps1
```

**Linux/macOS (Bash):**
```bash
./scripts/run_multi_gpu_tests.sh
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench --features gpu

# Run specific benchmark group
cargo bench --features gpu bench_memory_transfers
cargo bench --features gpu bench_cuda_kernels
```

## Test Configuration

### Environment Variables

- `GENERATE_JSON_REPORT=true` - Generate CI/CD-friendly JSON reports
- `ENABLE_SANITIZERS=true` - Run with address sanitizer (Linux only)

### GPU Requirements

**Single GPU Tests:**
- Basic memory allocation and kernel execution
- Single-device execution paths
- Error handling scenarios

**Multi-GPU Tests:**
- P2P memory transfers (requires P2P capable GPUs)
- Topology detection and optimal placement
- Load balancing across multiple devices
- Cross-device synchronization

### CUDA Requirements

- CUDA Toolkit 11.0+ recommended
- NVIDIA driver supporting target GPU architecture
- P2P support for multi-GPU tests (NVLink preferred)

## Test Categories

### 🔧 Unit Tests
- **Purpose**: Validate individual component functionality
- **Scope**: Single functions, small modules
- **Runtime**: Fast (< 1 minute total)
- **Dependencies**: Minimal GPU requirements

### 🔗 Integration Tests
- **Purpose**: Test component interactions
- **Scope**: Multi-component workflows
- **Runtime**: Medium (2-5 minutes)
- **Dependencies**: Full GPU stack

### ⚡ Performance Benchmarks
- **Purpose**: Measure and track performance
- **Scope**: End-to-end performance scenarios
- **Runtime**: Long (10-30 minutes)
- **Dependencies**: Multiple GPUs recommended

### 🛡️ Reliability Tests
- **Purpose**: Stress testing and error scenarios
- **Scope**: Edge cases, error recovery
- **Runtime**: Variable (depends on scenario)
- **Dependencies**: Various GPU configurations

## Interpreting Results

### Test Outcomes

- ✅ **Passed**: Test executed successfully
- ⚠️ **Skipped**: Test skipped (missing dependencies)
- ❌ **Failed**: Test failed (requires investigation)

### Common Skip Reasons

1. **No GPU Available**: Software-only environment
2. **Single GPU Only**: Multi-GPU tests skipped
3. **No P2P Support**: Peer-to-peer tests skipped
4. **CUDA Not Available**: CUDA-specific tests skipped
5. **Insufficient Memory**: Large memory tests skipped

### Performance Metrics

Benchmarks track:
- **Throughput**: Operations per second
- **Latency**: Time per operation
- **Memory Bandwidth**: GB/s transfer rates
- **GPU Utilization**: Compute efficiency
- **Power Consumption**: Energy usage (when available)

### Regression Detection

Performance regression tests fail if:
- Execution time exceeds baseline by >20%
- Memory usage increases by >10%
- Throughput drops below 90% of baseline

## Troubleshooting

### Common Issues

**GPU Not Detected:**
```bash
# Check GPU visibility
nvidia-smi
# Verify CUDA installation
nvcc --version
```

**P2P Tests Failing:**
```bash
# Check P2P capability
nvidia-smi topo -m
```

**Memory Allocation Failures:**
- Reduce test data sizes
- Close other GPU applications
- Check available GPU memory: `nvidia-smi`

**Compilation Errors:**
- Ensure CUDA Toolkit is installed
- Verify architecture compatibility
- Check include paths and libraries

### Debug Modes

**Verbose Output:**
```bash
cargo test --features gpu -- --nocapture
```

**Single-threaded Execution:**
```bash
cargo test --features gpu -- --test-threads=1
```

**Specific Test:**
```bash
cargo test --features gpu test_specific_function -- --exact
```

## Continuous Integration

### CI/CD Integration

The test framework generates machine-readable reports:

```bash
# Generate JSON report
GENERATE_JSON_REPORT=true ./scripts/run_multi_gpu_tests.sh
```

Output format:
```json
{
    "timestamp": "2023-01-01T00:00:00Z",
    "system": {
        "os": "Linux",
        "arch": "x86_64",
        "gpu_count": 2,
        "gpu_available": true
    },
    "results": {
        "unit_tests": true,
        "integration_tests": true,
        "benchmarks": true,
        "multi_gpu_tests": true
    }
}
```

### Automated Testing

Recommended CI pipeline:

1. **Pre-commit**: Unit tests only
2. **PR Validation**: Integration tests
3. **Nightly Builds**: Full benchmark suite
4. **Release Candidates**: Comprehensive validation

### Performance Monitoring

Set up automated performance tracking:
- Baseline measurements for each release
- Trend analysis for performance regressions
- Alert thresholds for significant changes

## Contributing

### Adding New Tests

1. **Unit Tests**: Add to `unit_tests.rs`
2. **Integration Tests**: Add to `multi_gpu_integration_tests.rs`
3. **Benchmarks**: Add to `../benches/multi_gpu_benchmarks.rs`

### Test Guidelines

- Use descriptive test names
- Include both success and failure paths
- Test edge cases and boundary conditions
- Provide clear error messages
- Use appropriate assertions
- Clean up resources properly

### Mock Data

For tests requiring specific GPU configurations:
- Use mock topology data for consistent testing
- Provide fallback paths for missing hardware
- Document hardware requirements clearly

## Performance Baselines

### Reference Performance

Typical performance on modern hardware:

**Memory Transfers:**
- System to GPU: ~12 GB/s (PCIe 4.0)
- GPU to GPU (P2P): ~50 GB/s (NVLink 3.0)
- GPU to System: ~12 GB/s (PCIe 4.0)

**Kernel Execution:**
- Simple kernels: <1ms launch overhead
- Complex kernels: Variable (workload dependent)
- Kernel compilation: 100-500ms (cached)

**Topology Discovery:**
- Initial discovery: 10-100ms
- Cached lookups: <1ms
- Placement planning: 1-10ms

### Optimization Targets

Performance goals for the framework:
- Memory allocation: <10ms for typical sizes
- P2P transfers: >80% of peak bandwidth
- Kernel launches: <100μs overhead
- Placement decisions: <5ms for complex topologies

---

*This testing framework ensures the reliability and performance of RONN's multi-GPU capabilities across diverse hardware configurations.*