#!/bin/bash
set -e

# Update package lists
apt-get update

# Install essential build tools
apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    lsb-release

# Add LLVM repository for clang-19
# wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
# add-apt-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-19 main"

# Update after adding repository
apt-get update

# Install clang-19, g++-14, and build tools
apt-get install -y \
    clang-19 \
    clang++-19 \
    lldb-19 \
    lld-19 \
    clangd-19 \
    g++-14 \
    gcc-14 \
    cmake \
    ninja-build \
    pkg-config \
    libc++-19-dev \
    libc++abi-19-dev

# Set up alternatives for clang
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100
update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100

# Set up alternatives for gcc
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

# Verify installations
echo "=== Installation Verification ==="
clang --version
g++ --version
cmake --version
ninja --version

echo "=== GPU Check ==="
nvidia-smi || echo "GPU not available in container setup"

echo "Setup complete!"