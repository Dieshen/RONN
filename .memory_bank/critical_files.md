# Critical File Locations

## ONNX Compatibility Layer

### Core ONNX Loading
- `crates/ronn-onnx/src/loader.rs:31` - Graph structure conversion from ONNX protobuf
- `crates/ronn-onnx/src/loader.rs:39` - Initializer loading (weights/constants)
- `crates/ronn-onnx/src/loader.rs:89` - Node attribute parsing and validation
- `crates/ronn-onnx/src/loader.rs:116` - Version compatibility checking
- `crates/ronn-onnx/src/types.rs` - Input/output shape inference and type system

### Operator Registry
- `crates/ronn-onnx/src/ops/mod.rs:44` - Operator registry with dynamic dispatch
- `crates/ronn-onnx/src/ops/` - Individual operator implementations

## Graph Optimization Pipeline

### Optimization Passes
- `crates/ronn-graph/src/passes/constant_folding.rs:8` - Constant folding pass
- `crates/ronn-graph/src/passes/dead_code.rs:8` - Dead code elimination
- `crates/ronn-graph/src/passes/fusion.rs:8` - Node fusion (Conv+BatchNorm+ReLU)
- `crates/ronn-graph/src/passes/layout.rs:8` - Layout optimization (NCHW/NHWC)
- `crates/ronn-graph/src/passes/provider_specific.rs:7` - CPU-specific optimizations
- `crates/ronn-graph/src/passes/provider_specific.rs:48` - GPU-specific optimizations

### Optimization Framework
- `crates/ronn-graph/src/optimizer.rs:8` - Optimization level framework (O0-O3)
- `crates/ronn-graph/src/optimizer.rs:96` - Iterative pass manager

## Brain-Inspired Features

### Hierarchical Reasoning Module (HRM)
- `crates/ronn-hrm/src/complexity.rs` - Complexity assessment engine (scores 0.0-1.0)
- `crates/ronn-hrm/src/complexity.rs:73` - assess() method with size/variance/dimensionality scoring
- `crates/ronn-hrm/src/router.rs` - Routing strategies and decision logic
- `crates/ronn-hrm/src/router.rs:70` - route() method for adaptive routing
- `crates/ronn-hrm/src/router.rs:103` - adaptive_route() implementation
- `crates/ronn-hrm/src/executor.rs` - System 1/System 2/Hybrid execution paths
- `crates/ronn-hrm/src/lib.rs` - Main HRM API and metrics tracking
- `examples/brain-features/src/main.rs` - Working demo showing all 3 routing paths

### Memory Systems (Implemented)
- `crates/ronn-memory/src/lib.rs` - Multi-tier memory system coordinator
- `crates/ronn-memory/src/working.rs` - Working memory implementation
- `crates/ronn-memory/src/episodic.rs` - Episodic memory with temporal indexing
- `crates/ronn-memory/src/semantic.rs` - Semantic memory knowledge graph
- `crates/ronn-memory/src/consolidation.rs` - Sleep consolidation engine

### Continual Learning (Implemented)
- `crates/ronn-learning/src/lib.rs` - Continual learning engine
- `crates/ronn-learning/src/timescales.rs` - Multi-timescale learning
- `crates/ronn-learning/src/ewc.rs` - Elastic Weight Consolidation
- `crates/ronn-learning/src/replay.rs` - Experience replay buffer

## Execution Providers

### Provider Framework
- `crates/ronn-providers/src/` - Base provider traits and registry
- `crates/ronn-providers/src/cpu/` - CPU provider implementation
- `crates/ronn-providers/src/gpu/` - GPU provider with Candle integration
- `crates/ronn-providers/src/bitnet/` - BitNet 1-bit quantization provider
- `crates/ronn-providers/src/wasm/` - WebAssembly provider
- `crates/ronn-providers/src/custom/` - Custom provider framework
- `crates/ronn-providers/src/custom/example_npu.rs` - Example NPU provider
- `crates/ronn-providers/src/custom/example_tpu.rs` - Example TPU provider

## High-Level API

### User-Facing API
- `crates/ronn-api/src/model.rs` - Model loading with builder pattern
- `crates/ronn-api/src/session.rs` - Session creation and management
- `crates/ronn-api/src/error.rs` - Structured error types
- `crates/ronn-api/src/` - Main API entry point

## Core Runtime

### Core Types and Operations
- `crates/ronn-core/src/` - Tensor types, session management
- `crates/ronn-core/` - Core runtime components

## Examples

### Working Examples
- `examples/simple-inference/src/main.rs` - Basic inference example
- `examples/brain-features/src/main.rs` - Brain-inspired features demonstration
- `examples/onnx-model/src/main.rs` - ONNX file loading example

## Documentation

### Design Documents
- `docs/brain_inspired_design.md:1` - Brain-inspired design specification
- `docs/execution_provider_design.md` - Execution provider architecture
- `docs/implementation_roadmap.md` - Implementation roadmap
- `docs/rust_ml_architecture.md` - Rust ML architecture document

### Project Documentation
- `README.md` - Project overview and quick start
- `README.md:103` - Performance tradeoffs table
- `TASKS.md` - Comprehensive development roadmap

## Build Configuration

### Workspace Configuration
- `Cargo.toml` - Workspace configuration
- `rust-toolchain.toml` - Rust toolchain version
- `rustfmt.toml` - Formatting configuration
- `clippy.toml` - Linting configuration

## CI/CD

### GitHub Actions
- `.github/workflows/ci.yml` - Main CI pipeline (build, test, lint)
- `.github/workflows/integration-tests.yml` - Integration tests with ONNX models
- `.github/workflows/benchmarks.yml` - Performance benchmarking

## Testing

### Test Files
- `crates/ronn-core/src/*/tests.rs` - Core component tests (62 tests)
- `crates/ronn-hrm/src/*/tests.rs` - HRM tests (22 tests)
- `crates/ronn-memory/src/*/tests.rs` - Memory system tests (13 tests)
- `crates/ronn-learning/src/*/tests.rs` - Learning engine tests (16 tests)
- `crates/ronn-providers/src/*/tests.rs` - Provider tests (62 tests)
- `crates/ronn-integration-tests/` - Integration tests with ONNX models

### Important Test Patterns
**Result Type in Tests**: Always use `type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;`
**Flexible Assertions**: Use `matches!()` for complexity levels that depend on input characteristics
