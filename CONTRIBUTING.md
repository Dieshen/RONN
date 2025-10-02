# Contributing to RONN

Thank you for your interest in contributing to RONN (Rust ONNX Neural Network Runtime)! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Running Benchmarks](#running-benchmarks)
- [Code Quality](#code-quality)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Development Setup

### Prerequisites

- Rust 1.75 or later
- Python 3.12 (for generating ONNX test fixtures)
- Git

### Clone and Build

```bash
git clone https://github.com/your-org/ronn.git
cd ronn
cargo build --workspace
```

### Generate Test Fixtures

RONN uses ONNX models as test fixtures. Generate them before running tests:

```bash
# Install Python dependencies
pip install onnx numpy

# Generate fixtures
cd crates/ronn-api/tests/fixtures
python create_fixtures.py
cd ../../..
```

## Running Tests

RONN has a comprehensive test suite with 797 tests covering all components.

### Run All Tests

```bash
# Generate fixtures first (if not already done)
cd crates/ronn-api/tests/fixtures && python create_fixtures.py && cd ../../..

# Run all workspace tests
cargo test --workspace
```

### Run Specific Crate Tests

```bash
# Core runtime tests
cargo test -p ronn-core

# API tests
cargo test -p ronn-api

# Provider tests
cargo test -p ronn-providers

# Graph optimization tests
cargo test -p ronn-graph

# ONNX parsing tests
cargo test -p ronn-onnx
```

### Run Doc Tests

Documentation examples are tested to ensure they compile and work correctly:

```bash
cargo test --doc --workspace
```

### Run Integration Tests

```bash
cargo test --test '*' -p ronn-api
```

### Run Specific Test

```bash
# Run a specific test by name
cargo test test_session_creation

# Run tests matching a pattern
cargo test tensor_
```

## Running Benchmarks

RONN uses Criterion.rs for benchmarking.

### Run All Benchmarks

```bash
cargo bench --all
```

### Run Specific Benchmark

```bash
# End-to-end inference benchmarks
cargo bench --bench end_to_end

# Comparative benchmarks (vs ONNX Runtime)
cargo bench --bench comparative
```

### View Benchmark Results

Benchmark results are stored in `target/criterion/`. Open the HTML reports:

```bash
# Linux/macOS
open target/criterion/report/index.html

# Windows
start target/criterion/report/index.html
```

## Code Quality

All contributions must pass the following checks:

### Format Code

```bash
cargo fmt --all
```

### Run Linter

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

### Check Compilation

```bash
cargo check --workspace --all-features
```

### Generate Coverage Report

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate HTML coverage report
cargo llvm-cov --workspace --html

# Generate LCOV format for CI
cargo llvm-cov --workspace --lcov --output-path lcov.info
```

Coverage reports are in `target/llvm-cov/html/index.html`.

**Coverage Target**: Maintain or improve code coverage. Project target is >80%.

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests**: All new features must include tests
   - Unit tests in the same file as the implementation
   - Integration tests in `tests/` directory
   - Doc tests for public API examples

3. **Generate fixtures** (if needed):
   ```bash
   cd crates/ronn-api/tests/fixtures
   python create_fixtures.py
   ```

4. **Run all tests**:
   ```bash
   cargo test --workspace
   cargo test --doc --workspace
   ```

5. **Check code quality**:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   ```

6. **Run benchmarks** (for performance-related changes):
   ```bash
   cargo bench --bench end_to_end
   ```

### Submitting the PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/benchmarks (if applicable)

3. **CI Requirements** - All PRs must pass:
   - ✅ All 797 tests on Linux, Windows, and macOS
   - ✅ Clippy with no warnings
   - ✅ Rustfmt check
   - ✅ Doc tests
   - ✅ Code coverage maintained or improved

### Review Process

- Maintainers will review your PR within 1-2 weeks
- Address review feedback by pushing new commits
- Once approved, a maintainer will merge your PR

## Project Structure

```
ronn/
├── crates/
│   ├── ronn-api/          # High-level API for model loading and inference
│   ├── ronn-core/         # Core runtime components (tensor, session, types)
│   ├── ronn-onnx/         # ONNX model parsing and protobuf handling
│   ├── ronn-graph/        # Graph optimization and transformation
│   ├── ronn-providers/    # Execution providers (CPU, GPU, WebAssembly, BitNet)
│   ├── ronn-memory/       # Memory management (placeholder)
│   └── ronn-learning/     # Learning components (placeholder)
├── benches/               # Benchmarks
│   ├── end_to_end.rs      # End-to-end inference benchmarks
│   └── comparative/       # Comparative benchmarks vs other runtimes
├── docs/                  # Documentation
└── .github/
    └── workflows/         # CI/CD workflows
        ├── ci.yml         # Main CI pipeline
        └── benchmarks.yml # Benchmark CI
```

## Development Guidelines

### Code Style

- Follow Rust idioms and best practices
- Use descriptive variable and function names
- Add documentation comments for public APIs
- Keep functions focused and small
- Prefer composition over inheritance

### Testing Philosophy

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test component interactions
- **Doc tests**: Ensure examples in documentation work
- **Property tests**: Use for complex algorithms (with proptest)

### Performance Considerations

- Profile before optimizing
- Use benchmarks to validate optimizations
- Document performance characteristics
- Consider memory allocation patterns

### Error Handling

- Use `Result<T, E>` for recoverable errors
- Use specific error types, not `anyhow::Error` in public APIs
- Provide helpful error messages
- Never silently ignore errors

### Documentation

- Document all public APIs with `///` comments
- Include examples in documentation
- Explain **why**, not just **what**
- Update README.md for major features

## Getting Help

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Chat**: Join our Discord/Slack (if available)

## License

By contributing to RONN, you agree that your contributions will be licensed under the same license as the project (MIT or Apache-2.0).

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.
