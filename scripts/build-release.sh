#!/bin/bash
set -euo pipefail

# RONN Release Build Script
# Builds RONN with maximum optimizations for production deployment

echo "🚀 Building RONN for release..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
cargo clean

# Set optimization flags
export RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1 -C embed-bitcode=yes"

# Build with maximum optimizations
echo "🏗️ Building with maximum optimizations..."
cargo build --workspace --release

# Run tests to ensure everything works
echo "🧪 Running release tests..."
cargo test --workspace --release

# Run benchmarks if available
echo "📊 Running benchmarks..."
if [ -f "benches/runtime_benchmarks.rs" ]; then
    cargo bench --workspace
else
    echo "⚠️ No benchmarks found, skipping..."
fi

# Check binary sizes
echo "📏 Binary sizes:"
find target/release -name "*.exe" -o -name "ronn*" -type f -executable | while read -r binary; do
    if [ -f "$binary" ]; then
        size=$(du -h "$binary" | cut -f1)
        echo "  $(basename "$binary"): $size"
    fi
done

# Strip symbols if possible (Unix-like systems)
if command -v strip &> /dev/null; then
    echo "🔪 Stripping debug symbols..."
    find target/release -name "*.exe" -o -name "ronn*" -type f -executable | while read -r binary; do
        if [ -f "$binary" ]; then
            strip "$binary" 2>/dev/null || true
        fi
    done
fi

# Create distribution directory
echo "📦 Preparing distribution..."
mkdir -p dist/

# Copy binaries
find target/release -name "*.exe" -o -name "ronn*" -type f -executable | while read -r binary; do
    if [ -f "$binary" ]; then
        cp "$binary" dist/
    fi
done

# Copy documentation
if [ -f "README.md" ]; then
    cp README.md dist/
fi

if [ -f "LICENSE" ]; then
    cp LICENSE dist/
fi

# Create tarball
if command -v tar &> /dev/null; then
    PLATFORM=$(uname -m)
    TARBALL="dist/ronn-${PLATFORM}.tar.gz"
    echo "📦 Creating tarball: $TARBALL"
    tar -czf "$TARBALL" -C dist .
    echo "✅ Created: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
fi

echo ""
echo "🎉 Release build complete!"
echo "📁 Files available in ./dist/"
echo ""

# Performance targets check
echo "🎯 Performance Targets:"
echo "  Binary Size: <50MB (inference only)"
echo "  Memory Usage: <4GB total"
echo "  Latency: <10ms P50, <30ms P95"
echo ""
echo "Run benchmarks to verify performance targets."