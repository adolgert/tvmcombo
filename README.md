# tvmcombo

This repository is a demonstration of how to add a neural net to a legacy application.

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
