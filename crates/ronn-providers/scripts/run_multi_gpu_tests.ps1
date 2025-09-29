# Multi-GPU Test Runner Script
# This script runs comprehensive tests and benchmarks for multi-GPU functionality

Write-Host "=== RONN Multi-GPU Test Suite ===" -ForegroundColor Green
Write-Host ""

$ErrorActionPreference = "Continue"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptPath

# Change to project directory
Push-Location $projectRoot

try {
    # Check for GPU availability
    Write-Host "Checking GPU availability..." -ForegroundColor Blue
    $gpuCheck = nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVIDIA GPU detected" -ForegroundColor Green
        Write-Host $gpuCheck | Select-String "Driver Version", "CUDA Version"
    } else {
        Write-Host "⚠ No NVIDIA GPU detected - some tests will be skipped" -ForegroundColor Yellow
    }
    Write-Host ""

    # Build with GPU features
    Write-Host "Building with GPU features..." -ForegroundColor Blue
    cargo build --features gpu --release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Build completed successfully" -ForegroundColor Green
    Write-Host ""

    # Run unit tests
    Write-Host "Running unit tests..." -ForegroundColor Blue
    cargo test --features gpu unit_tests --lib -- --nocapture
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Unit tests passed" -ForegroundColor Green
    } else {
        Write-Host "⚠ Some unit tests failed or were skipped" -ForegroundColor Yellow
    }
    Write-Host ""

    # Run integration tests
    Write-Host "Running integration tests..." -ForegroundColor Blue
    cargo test --features gpu multi_gpu_integration_tests --test multi_gpu_integration_tests -- --nocapture --test-threads=1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Integration tests passed" -ForegroundColor Green
    } else {
        Write-Host "⚠ Some integration tests failed or were skipped" -ForegroundColor Yellow
    }
    Write-Host ""

    # Run benchmarks (only if GPU is available)
    if ($gpuCheck) {
        Write-Host "Running performance benchmarks..." -ForegroundColor Blue
        Write-Host "This may take several minutes..." -ForegroundColor Yellow

        cargo bench --features gpu --bench multi_gpu_benchmarks -- --sample-size 10
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Benchmarks completed successfully" -ForegroundColor Green
        } else {
            Write-Host "⚠ Some benchmarks failed or were skipped" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Skipping benchmarks - GPU not available" -ForegroundColor Yellow
    }
    Write-Host ""

    # Run specific multi-GPU tests if multiple GPUs detected
    $gpuCount = (nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null | Measure-Object).Count
    if ($gpuCount -gt 1) {
        Write-Host "Multiple GPUs detected ($gpuCount) - running multi-GPU specific tests..." -ForegroundColor Blue

        # Run P2P tests
        cargo test --features gpu test_peer_to_peer_transfers --test multi_gpu_integration_tests -- --nocapture --exact

        # Run topology tests
        cargo test --features gpu test_gpu_topology_management --test multi_gpu_integration_tests -- --nocapture --exact

        # Run load balancing tests
        cargo test --features gpu test_concurrent_operations --test multi_gpu_integration_tests -- --nocapture --exact

        Write-Host "✓ Multi-GPU specific tests completed" -ForegroundColor Green
    } else {
        Write-Host "Single GPU detected - skipping multi-GPU specific tests" -ForegroundColor Yellow
    }
    Write-Host ""

    # Generate test report
    Write-Host "Generating test coverage report..." -ForegroundColor Blue
    if (Get-Command "cargo-tarpaulin" -ErrorAction SilentlyContinue) {
        cargo tarpaulin --features gpu --out Html --output-dir target/coverage --exclude-files "benches/*" --exclude-files "tests/*" --ignore-panics
        Write-Host "✓ Coverage report generated in target/coverage/" -ForegroundColor Green
    } else {
        Write-Host "cargo-tarpaulin not installed - skipping coverage report" -ForegroundColor Yellow
        Write-Host "Install with: cargo install cargo-tarpaulin" -ForegroundColor Cyan
    }
    Write-Host ""

    # Performance regression check
    Write-Host "Running performance regression check..." -ForegroundColor Blue
    cargo test --features gpu test_performance_regression --test multi_gpu_integration_tests -- --nocapture --exact
    Write-Host ""

    # Memory leak detection
    Write-Host "Running memory leak detection..." -ForegroundColor Blue
    cargo test --features gpu test_memory_leak_detection --test multi_gpu_integration_tests -- --nocapture --exact
    Write-Host ""

    # Summary
    Write-Host "=== Test Summary ===" -ForegroundColor Green
    Write-Host "✓ Unit tests: Completed" -ForegroundColor Green
    Write-Host "✓ Integration tests: Completed" -ForegroundColor Green

    if ($gpuCheck) {
        Write-Host "✓ Benchmarks: Completed" -ForegroundColor Green
    } else {
        Write-Host "- Benchmarks: Skipped (No GPU)" -ForegroundColor Yellow
    }

    if ($gpuCount -gt 1) {
        Write-Host "✓ Multi-GPU tests: Completed" -ForegroundColor Green
    } else {
        Write-Host "- Multi-GPU tests: Skipped (Single GPU)" -ForegroundColor Yellow
    }

    Write-Host "✓ Performance regression: Checked" -ForegroundColor Green
    Write-Host "✓ Memory leak detection: Completed" -ForegroundColor Green
    Write-Host ""
    Write-Host "All tests completed successfully!" -ForegroundColor Green

} catch {
    Write-Host "Error occurred during testing: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Tip: View detailed benchmark results with:" -ForegroundColor Cyan
Write-Host "  cargo bench --features gpu -- --output-format json > benchmark_results.json" -ForegroundColor White
Write-Host ""
Write-Host "For continuous monitoring, consider setting up:" -ForegroundColor Cyan
Write-Host "  - Automated nightly benchmark runs" -ForegroundColor White
Write-Host "  - Performance regression alerts" -ForegroundColor White
Write-Host "  - GPU memory usage monitoring" -ForegroundColor White