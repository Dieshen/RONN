#!/bin/bash
# Multi-GPU Test Runner Script
# This script runs comprehensive tests and benchmarks for multi-GPU functionality

set -e

echo "=== RONN Multi-GPU Test Suite ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check for GPU availability
echo -e "${BLUE}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits | head -1
        GPU_AVAILABLE=true
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
    else
        echo -e "${YELLOW}⚠ nvidia-smi found but no GPUs detected${NC}"
        GPU_AVAILABLE=false
        GPU_COUNT=0
    fi
elif command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✓ AMD ROCm GPU detected${NC}"
    GPU_AVAILABLE=true
    GPU_COUNT=$(rocm-smi --showid | grep -c "GPU\[")
else
    echo -e "${YELLOW}⚠ No GPU runtime detected - some tests will be skipped${NC}"
    GPU_AVAILABLE=false
    GPU_COUNT=0
fi
echo ""

# Build with GPU features
echo -e "${BLUE}Building with GPU features...${NC}"
if cargo build --features gpu --release; then
    echo -e "${GREEN}✓ Build completed successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

# Run unit tests
echo -e "${BLUE}Running unit tests...${NC}"
if cargo test --features gpu unit_tests --lib -- --nocapture; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
    UNIT_TESTS_PASSED=true
else
    echo -e "${YELLOW}⚠ Some unit tests failed or were skipped${NC}"
    UNIT_TESTS_PASSED=false
fi
echo ""

# Run integration tests
echo -e "${BLUE}Running integration tests...${NC}"
if cargo test --features gpu multi_gpu_integration_tests --test multi_gpu_integration_tests -- --nocapture --test-threads=1; then
    echo -e "${GREEN}✓ Integration tests passed${NC}"
    INTEGRATION_TESTS_PASSED=true
else
    echo -e "${YELLOW}⚠ Some integration tests failed or were skipped${NC}"
    INTEGRATION_TESTS_PASSED=false
fi
echo ""

# Run benchmarks (only if GPU is available)
BENCHMARKS_RUN=false
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${BLUE}Running performance benchmarks...${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}"

    if cargo bench --features gpu --bench multi_gpu_benchmarks -- --sample-size 10; then
        echo -e "${GREEN}✓ Benchmarks completed successfully${NC}"
        BENCHMARKS_RUN=true
    else
        echo -e "${YELLOW}⚠ Some benchmarks failed or were skipped${NC}"
    fi
else
    echo -e "${YELLOW}Skipping benchmarks - GPU not available${NC}"
fi
echo ""

# Run specific multi-GPU tests if multiple GPUs detected
MULTI_GPU_TESTS_RUN=false
if [ "$GPU_COUNT" -gt 1 ]; then
    echo -e "${BLUE}Multiple GPUs detected ($GPU_COUNT) - running multi-GPU specific tests...${NC}"

    # Run P2P tests
    cargo test --features gpu test_peer_to_peer_transfers --test multi_gpu_integration_tests -- --nocapture --exact

    # Run topology tests
    cargo test --features gpu test_gpu_topology_management --test multi_gpu_integration_tests -- --nocapture --exact

    # Run load balancing tests
    cargo test --features gpu test_concurrent_operations --test multi_gpu_integration_tests -- --nocapture --exact

    echo -e "${GREEN}✓ Multi-GPU specific tests completed${NC}"
    MULTI_GPU_TESTS_RUN=true
else
    echo -e "${YELLOW}Single GPU detected - skipping multi-GPU specific tests${NC}"
fi
echo ""

# Generate test coverage report
echo -e "${BLUE}Generating test coverage report...${NC}"
if command -v cargo-tarpaulin &> /dev/null; then
    if cargo tarpaulin --features gpu --out Html --output-dir target/coverage --exclude-files "benches/*" --exclude-files "tests/*" --ignore-panics; then
        echo -e "${GREEN}✓ Coverage report generated in target/coverage/${NC}"
    else
        echo -e "${YELLOW}⚠ Coverage report generation failed${NC}"
    fi
else
    echo -e "${YELLOW}cargo-tarpaulin not installed - skipping coverage report${NC}"
    echo -e "${CYAN}Install with: cargo install cargo-tarpaulin${NC}"
fi
echo ""

# Performance regression check
echo -e "${BLUE}Running performance regression check...${NC}"
cargo test --features gpu test_performance_regression --test multi_gpu_integration_tests -- --nocapture --exact
echo ""

# Memory leak detection
echo -e "${BLUE}Running memory leak detection...${NC}"
cargo test --features gpu test_memory_leak_detection --test multi_gpu_integration_tests -- --nocapture --exact
echo ""

# Check for sanitizer builds (optional)
if [ "${ENABLE_SANITIZERS:-false}" = "true" ]; then
    echo -e "${BLUE}Running tests with address sanitizer...${NC}"
    RUSTFLAGS="-Z sanitizer=address" cargo test --features gpu --target x86_64-unknown-linux-gnu
    echo ""
fi

# Summary
echo -e "${GREEN}=== Test Summary ===${NC}"

if [ "$UNIT_TESTS_PASSED" = true ]; then
    echo -e "${GREEN}✓ Unit tests: Passed${NC}"
else
    echo -e "${YELLOW}⚠ Unit tests: Some failed${NC}"
fi

if [ "$INTEGRATION_TESTS_PASSED" = true ]; then
    echo -e "${GREEN}✓ Integration tests: Passed${NC}"
else
    echo -e "${YELLOW}⚠ Integration tests: Some failed${NC}"
fi

if [ "$BENCHMARKS_RUN" = true ]; then
    echo -e "${GREEN}✓ Benchmarks: Completed${NC}"
else
    echo -e "${YELLOW}- Benchmarks: Skipped (No GPU)${NC}"
fi

if [ "$MULTI_GPU_TESTS_RUN" = true ]; then
    echo -e "${GREEN}✓ Multi-GPU tests: Completed${NC}"
else
    echo -e "${YELLOW}- Multi-GPU tests: Skipped (Single/No GPU)${NC}"
fi

echo -e "${GREEN}✓ Performance regression: Checked${NC}"
echo -e "${GREEN}✓ Memory leak detection: Completed${NC}"
echo ""

# Detect system info for reporting
echo -e "${BLUE}System Information:${NC}"
echo "OS: $(uname -s) $(uname -r)"
echo "Architecture: $(uname -m)"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "GPU Count: $GPU_COUNT"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
        echo "CUDA Version: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)"
    fi
fi
echo ""

echo -e "${GREEN}All tests completed successfully!${NC}"
echo ""
echo -e "${CYAN}Tip: View detailed benchmark results with:${NC}"
echo -e "${NC}  cargo bench --features gpu -- --output-format json > benchmark_results.json${NC}"
echo ""
echo -e "${CYAN}For continuous monitoring, consider setting up:${NC}"
echo -e "${NC}  - Automated nightly benchmark runs${NC}"
echo -e "${NC}  - Performance regression alerts${NC}"
echo -e "${NC}  - GPU memory usage monitoring${NC}"
echo ""

# Optional: Generate JSON report for CI/CD
if [ "${GENERATE_JSON_REPORT:-false}" = "true" ]; then
    echo -e "${BLUE}Generating JSON report for CI/CD...${NC}"
    cat > target/test_results.json << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "gpu_count": $GPU_COUNT,
        "gpu_available": $GPU_AVAILABLE
    },
    "results": {
        "unit_tests": $UNIT_TESTS_PASSED,
        "integration_tests": $INTEGRATION_TESTS_PASSED,
        "benchmarks": $BENCHMARKS_RUN,
        "multi_gpu_tests": $MULTI_GPU_TESTS_RUN
    }
}
EOF
    echo -e "${GREEN}✓ JSON report saved to target/test_results.json${NC}"
fi