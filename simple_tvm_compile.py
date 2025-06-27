#!/usr/bin/env python3
"""
Simple TVM compilation using the relax frontend for ONNX models.
This is a more direct approach to compile ONNX to CUDA kernels.
"""

import os
import sys
import numpy as np

# Add TVM python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tvm', 'python'))

# Set environment variable for TVM library path
os.environ['TVM_LIBRARY_PATH'] = os.path.join(os.path.dirname(__file__), 'tvm', 'build')

try:
    import tvm
    from tvm import relax
    from tvm.relax.frontend import onnx as onnx_frontend
    import onnx
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTVM needs to be built first. Run these commands:")
    print("cd tvm && mkdir -p build && cd build")
    print("cmake .. -DUSE_CUDA=ON -DUSE_LLVM=ON")
    print("make -j$(nproc)")
    sys.exit(1)

def compile_onnx_to_cuda():
    """Main compilation function."""
    # First ensure we have an ONNX model
    onnx_path = "neural_net.onnx"
    if not os.path.exists(onnx_path):
        print("Creating ONNX model first...")
        os.system("python3 create_neural_net.py")
    
    print(f"Loading ONNX model from {onnx_path}...")
    onnx_model = onnx.load(onnx_path)
    
    # Convert to Relax IR
    print("Converting ONNX to TVM Relax IR...")
    mod = onnx_frontend.from_onnx(onnx_model, keep_params_in_input=False)
    
    # Define target
    target = tvm.target.cuda(arch="sm_89")
    
    # Apply memory binding and scheduling passes
    print("Applying memory binding optimizations...")
    with tvm.target.Target(target):
        mod = tvm.tir.transform.StorageRewrite()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
    
    print("Building module for CUDA...")
    # Apply optimization passes before building
    # with tvm.transform.PassContext(opt_level=3):
    #     ex = relax.build(mod, target, runtime=tvm.runtime.cuda())
    ex = tvm.compile(mod, target)    
    # Export as shared library
    output_path = "tvm_neural_net.so"
    print(f"Exporting to {output_path}...")
    ex.export_library(output_path)
    
    print(f"\nSuccess! Compiled model saved to {output_path}")
    print("\nIntegration summary:")
    print("- Input format: ONNX")
    print("- Intermediate format: TVM Relax IR") 
    print("- Output: Compiled shared library (.so)")
    print("- Target: NVIDIA CUDA sm_75")
    
    # Generate simple integration example
    generate_integration_example()

def generate_integration_example():
    """Generate a simple C++ integration example."""
    cpp_code = """// Simple TVM Runtime Integration Example
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/device_api.h>

class TVMModel {
private:
    tvm::runtime::Module mod;
    tvm::runtime::PackedFunc f_init;
    tvm::runtime::PackedFunc f_run;
    
public:
    void Load(const std::string& lib_path) {
        // Load the compiled module
        mod = tvm::runtime::Module::LoadFromFile(lib_path);
        
        // Get the entry functions
        f_init = mod.GetFunction("_initialize");
        f_run = mod.GetFunction("main");
    }
    
    void Run(float* input_data, float* output_data) {
        // Create GPU context
        DLDevice dev{kDLCUDA, 0};
        
        // Create input array
        tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(
            {1, 4}, DLDataType{kDLFloat, 32, 1}, dev);
        input.CopyFromBytes(input_data, 4 * sizeof(float));
        
        // Run inference
        tvm::runtime::NDArray output = f_run(input);
        
        // Copy output back
        output.CopyToBytes(output_data, 2 * sizeof(float));
    }
};
"""
    
    with open("tvm_simple_integration.cpp", "w") as f:
        f.write(cpp_code)
    print("\nGenerated C++ integration example: tvm_simple_integration.cpp")

if __name__ == "__main__":
    compile_onnx_to_cuda()