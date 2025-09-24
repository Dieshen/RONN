#!/bin/bash
set -euo pipefail

# RONN Development Environment Setup Script
# This script sets up the development environment for the RONN project

echo "🦀 Setting up RONN development environment..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check Rust version
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
REQUIRED_VERSION="1.90.0"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "❌ Rust version $REQUIRED_VERSION or higher is required. Found: $RUST_VERSION"
    echo "Run: rustup update"
    exit 1
fi

echo "✅ Rust version $RUST_VERSION detected"

# Install required components
echo "📦 Installing Rust components..."
rustup component add rustfmt clippy llvm-tools-preview

# Install additional tools
echo "🔧 Installing development tools..."
cargo install --locked cargo-llvm-cov || echo "cargo-llvm-cov already installed"
cargo install --locked cargo-deny || echo "cargo-deny already installed"
cargo install --locked cargo-audit || echo "cargo-audit already installed"
cargo install --locked cargo-outdated || echo "cargo-outdated already installed"

# Set up git hooks if .git exists
if [ -d ".git" ]; then
    echo "⚙️ Setting up git hooks..."

    # Create pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Format check
cargo fmt --all -- --check
if [ $? -ne 0 ]; then
    echo "❌ Code formatting issues detected. Run 'cargo fmt --all' to fix."
    exit 1
fi

# Clippy check
cargo clippy --workspace --all-targets -- -D warnings
if [ $? -ne 0 ]; then
    echo "❌ Clippy warnings detected. Please fix them."
    exit 1
fi

# Quick tests
cargo test --workspace --lib
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix them."
    exit 1
fi

echo "✅ Pre-commit checks passed"
EOF

    chmod +x .git/hooks/pre-commit
    echo "✅ Git pre-commit hooks installed"
else
    echo "⚠️ Not a git repository, skipping git hooks setup"
fi

# Build the project
echo "🏗️ Building project..."
cargo build --workspace

# Run tests
echo "🧪 Running tests..."
cargo test --workspace

# Generate documentation
echo "📚 Generating documentation..."
cargo doc --workspace --no-deps

echo ""
echo "🎉 RONN development environment setup complete!"
echo ""
echo "Available commands:"
echo "  ./scripts/build-release.sh  - Build optimized release"
echo "  ./scripts/test.sh          - Run all tests"
echo "  ./scripts/bench.sh         - Run benchmarks"
echo "  ./scripts/docs.sh          - Generate and open documentation"
echo ""
echo "Happy coding! 🦀"