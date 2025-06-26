#!/usr/bin/env python3
"""
Compile ONNX model to TVM for NVIDIA GPU deployment.
This script takes the ONNX model created by create_neural_net.py and compiles it
using Apache TVM to generate optimized CUDA kernels.
"""

import os
import numpy as np
import tvm
from tvm import relay
import onnx

def load_onnx_model(model_path):
    """Load ONNX model and convert to TVM Relay IR."""
    print(f"Loading ONNX model from {model_path}...")
    onnx_model = onnx.load(model_path)
    
    # Define input shape - batch size 1, 4 features
    input_name = "input"
    shape_dict = {input_name: (1, 4)}
    
    # Convert ONNX to Relay IR
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    print("Model successfully converted to TVM Relay IR")
    return mod, params, input_name

def compile_model_for_cuda(mod, params, target_arch="sm_75"):
    """
    Compile the model for CUDA execution.
    
    Args:
        mod: TVM relay module
        params: Model parameters
        target_arch: CUDA compute capability (default: sm_75 for RTX 2060)
    
    Returns:
        Compiled TVM module
    """
    print(f"Compiling model for CUDA target: {target_arch}")
    
    # Define target - CUDA with specific architecture
    target = tvm.target.cuda(arch=target_arch)
    
    # Build the module
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    print("Model compilation completed successfully")
    return lib

def export_compiled_module(lib, export_path="tvm_neural_net.so"):
    """Export the compiled module as a shared library."""
    print(f"Exporting compiled module to {export_path}...")
    lib.export_library(export_path)
    print(f"Module exported successfully to {export_path}")
    
def generate_cpp_integration_code():
    """Generate example C++ code for integrating the TVM module."""
    cpp_code = '''// Example C++ code for loading and running the TVM-compiled model
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <cuda_runtime.h>

class TVMNeuralNet {
private:
    tvm::runtime::Module mod_;
    tvm::runtime::PackedFunc set_input_;
    tvm::runtime::PackedFunc run_;
    tvm::runtime::PackedFunc get_output_;
    DLDevice dev_{kDLCUDA, 0};
    
public:
    void Load(const std::string& lib_path) {
        // Load the compiled module
        mod_ = tvm::runtime::Module::LoadFromFile(lib_path);
        
        // Get the module functions
        auto gmod = mod_.GetFunction("default");
        set_input_ = gmod.GetFunction("set_input");
        run_ = gmod.GetFunction("run");
        get_output_ = gmod.GetFunction("get_output");
    }
    
    void Run(float* input_data, float* output_data) {
        // Create input tensor
        DLTensor input;
        input.data = input_data;
        input.device = dev_;
        input.ndim = 2;
        int64_t shape[2] = {1, 4};
        input.shape = shape;
        input.dtype = {kDLFloat, 32, 1};
        input.strides = nullptr;
        input.byte_offset = 0;
        
        // Set input
        set_input_("input", &input);
        
        // Run inference
        run_();
        
        // Get output
        tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
        tvm::runtime::NDArray output = get_output(0);
        
        // Copy output to host if needed
        output.CopyToBytes(output_data, 2 * sizeof(float));
    }
};
'''
    
    with open("tvm_integration_example.cpp", "w") as f:
        f.write(cpp_code)
    print("Generated C++ integration example: tvm_integration_example.cpp")

def test_compiled_model(lib, input_name):
    """Test the compiled model with sample input."""
    print("\nTesting compiled model...")
    
    # Create runtime module
    dev = tvm.cuda(0)
    module = lib["default"](dev)
    
    # Prepare test input
    test_input = np.array([[3.34286e+06, 3186.6, 5000.0, 1250.0]], dtype=np.float32)
    
    # Set input
    module.set_input(input_name, test_input)
    
    # Run inference
    module.run()
    
    # Get output
    output = module.get_output(0).numpy()
    
    print(f"Test input: {test_input}")
    print(f"Model output: {output}")
    print(f"Predicted class: {np.argmax(output)}")

def main():
    # First, ensure we have an ONNX model
    onnx_path = "neural_net.onnx"
    if not os.path.exists(onnx_path):
        print("ONNX model not found. Creating it first...")
        os.system("python3 create_neural_net.py")
    
    # Load ONNX model
    mod, params, input_name = load_onnx_model(onnx_path)
    
    # Compile for CUDA
    lib = compile_model_for_cuda(mod, params)
    
    # Export as shared library
    export_compiled_module(lib)
    
    # Generate C++ integration code
    generate_cpp_integration_code()
    
    # Test the compiled model
    test_compiled_model(lib, input_name)
    
    print("\n=== TVM Compilation Summary ===")
    print("Input format: ONNX")
    print("Intermediate format: TVM Relay IR")
    print("Output format: Compiled shared library (.so)")
    print("Target: NVIDIA CUDA (sm_75)")
    print("Integration: Link tvm_neural_net.so with your C++ application")
    print("===============================")

if __name__ == "__main__":
    main()