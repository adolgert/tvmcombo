# tvmcombo

## Install and Run

Clang-14's commandline says its newest supported CUDA version is 11.5.

 1. Look at BUILD_TVM.md to install TVM. It's a bear.

Open a terminal in the Docker and run:
```
conda activate tvm-build-venv
export TVMHOME=/workspaces/tvmcombo/tvm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TVMHOME/build:$TVMHOME/build/lib:$TVMHOME/onnxruntime-linux-x64-1.20.1/lib:$TVMHOME/python/build/lib.linux-x86_64-cpython-311/tvm/ffi:$TVMHOME/python/tvm/ffi
```
Install the already-build TVM.
```
INSTALL_DEV=ON sudo cmake --install .
```
Sometimes it helps to preload the tvm.so file in a Python script.
```
ctypes.CDLL('/workspaces/tvmcombo/tvm/build/libtvm.so', ctypes.RTLD_GLOBAL)
```
 1. Run python create_neural_net.py to make an ONNX.

 1. Run python simple_tvm_compile.py  to convert that to TVM's .so file.


This repository is a demonstration of how to add a neural net to a legacy application.

## Python Files Summary

- **create_neural_net.py**: Creates a simple feedforward neural network with 4→16→8→2 architecture using PyTorch, designed to analyze distribution integration results. The script exports the model to ONNX format and includes testing functionality with sample input data.

- **test_tvm_compile.py**: Tests TVM installation and basic functionality by checking module availability, CUDA support, and performing simple compilation tests. It provides diagnostic information and recommended next steps for TVM workflow setup.

- **tvm_codegen_approach.py**: Generates optimized CUDA kernel code from PyTorch models using TVM-style optimization patterns including shared memory usage and coalesced memory access. Creates complete C++ wrapper classes and binary weight files for integration with existing CUDA applications.

- **z.py**: A minimal TVM core module loading test script that adds the TVM Python path and imports the core FFI module. Provides basic verification that TVM's Python bindings are correctly installed and accessible.

- **compile_onnx_to_tvm.py**: Compiles ONNX neural network models to optimized TVM CUDA kernels using the relax frontend and exports them as shared libraries. Includes comprehensive C++ integration examples and testing functionality for the compiled models.

- **simple_tvm_compile.py**: Provides a streamlined approach to compile ONNX models to CUDA using TVM's relax frontend with optimization passes and memory binding. Generates both the compiled shared library and C++ integration code for easy deployment in existing applications.

## Legacy Application

  Target Architecture:
  - NVIDIA Compute Capability 7.5 (sm_75) - optimized for RTX 2060 GPU

  Compiler:
  - Clang++ for both host and device code compilation (not NVIDIA's nvcc)
  - C++17 standard
  - CMake build system with CUDAToolkit integration

  GPU:
  - NVIDIA GeForce RTX 2060
  - Processes 1,048,576 distributions using 4,096 blocks × 256 threads per block

  Memory Model:
  - Unified Memory (cudaMallocManaged) - allows seamless data access between CPU and GPU
  - Default CUDA Stream (stream 0) - blocking stream that synchronizes all GPU operations
  - Data flow: Host → Unified Memory → GPU computation → Host result aggregation

  Computation:
  - Integrates four probability distribution types: Gamma, Exponential, Weibull, and LogLogistic
  - Each thread processes one distribution using numerical integration (1,000 steps)
  - Uses cudaDeviceSynchronize() for explicit GPU-CPU synchronization
  - Performance: ~318ms GPU execution time, ~910ms total runtime

  This represents a typical legacy GPU application that will be enhanced with Apache TVM neural network capabilities running on separate CUDA streams.

## Addition of Neural Net

Followed this instruction to install TVM: https://tvm.apache.org/docs/install/from_source.html
Use `conda activate tvm-build-venv`.

  Legacy Application (RTX 2060, sm_75, clang CUDA compilation):
  - Processes 1M+ probability distributions (Gamma, Exponential, Weibull, LogLogistic)
  - Uses unified memory and default CUDA stream
  - ~39μs GPU execution time for distribution integrations

  Neural Network Integration:
  - Created custom CUDA neural network implementation (4→16→8→2 architecture)
  - Separate CUDA streams for legacy computation and neural inference
  - CUDA events for proper synchronization between streams
  - Takes distribution statistics as input: [total_integral, mean_integral, max_integral, std_integral]
  - ~102μs neural inference execution time

  Key Technical Achievements:
  1. Stream Isolation: Legacy computation runs on separate stream from neural inference
  2. Event Synchronization: Proper coordination using CUDA events
  3. Conservative Implementation: Used custom CUDA kernels instead of complex TVM integration
  4. Clang Compatibility: Maintained clang++ CUDA compilation throughout

  Performance:
  - Total execution time: ~877ms
  - Legacy computation: ~39μs
  - Neural inference: ~102μs
  - Both running concurrently without blocking

  The integration demonstrates how to add modern ML capabilities to legacy GPU applications while maintaining the existing architecture and compilation
   approach.
   
## The example application

We need to calculate an example application. That application will create a vector
of distributions including Gamma, Exponential, Weibull, and LogLogistic distributions
with default parameters. It will be a long vector so that it uses all GPU resources.
The calculation will integrate the cumulative distributions from time 0 to t and return the
sum of the integrals.
