#!/bin/bash
set -e

echo "=== Building TVM Combo Legacy Application ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -G Ninja

# Build
echo "Building..."
ninja

echo "=== Build Complete ==="
echo "Run './build/legacy_app' to execute the application"