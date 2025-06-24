# tvmcombo

This repository is a demonstration of how to add a neural net to a legacy application.

The legacy application is:

 * written in C++.
 * uses the GPU to calculate the integral of a vector of probability distribution functions.
 * targets an NVIDIA GPU but does NOT use the nvidia compiler. It uses clang which can target NVIDIA GPUs.
 * uses Unified Memory to call the GPU.
 * uses the Default Stream (usually stream 7 for NVIDIA) which is a blocking stream for all other computations.
 * uses a simple for-each call to start the GPU calculations.

The addition to this legacy application will create a simple neural net:

 * Using Apache TVM.
 * Executing the neural net on a specific stream (not the default GPU stream).
 * Using events to coordinate with the legacy calculation.

The build system is CMake.

## The example application

We need to calculate an example application. That application will create a vector
of distributions including Gamma, Exponential, Weibull, and LogLogistic distributions
with default parameters. It will be a long vector so that it uses all GPU resources.
The calculation will integrate the cumulative distributions from time 0 to t and return the
sum of the integrals.
