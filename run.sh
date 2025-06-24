#!/bin/bash
set -e

echo "=== Running TVM Combo Legacy Application ==="

# Check if binary exists
if [ ! -f "build/legacy_app" ]; then
    echo "Binary not found. Building first..."
    ./build.sh
fi

# Run the application
echo "Executing legacy application..."
./build/legacy_app

echo "=== Execution Complete ==="