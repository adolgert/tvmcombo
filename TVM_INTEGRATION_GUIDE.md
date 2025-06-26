# TVM Integration Guide for CUDA Applications

This guide explains how to integrate TVM-compiled neural networks into existing CUDA applications.

## Overview

Apache TVM provides several approaches for model compilation and runtime integration:

### 1. **Intermediate Formats**

TVM supports multiple input formats:
- **ONNX** (recommended): Universal format, well-supported
- **PyTorch**: Via torch.export and TVM's PyTorch frontend
- **TensorFlow/Keras**: Via SavedModel format
- **Relay IR**: TVM's older graph-level IR (being phased out)
- **Relax IR**: TVM's newer unified IR (recommended for new projects)

### 2. **Compilation Approaches**

#### Option A: Static Compilation (Recommended)
- Compile model ahead-of-time to a shared library (.so)
- Best performance, minimal runtime overhead
- Easiest C++ integration

#### Option B: TVM Module Format
- Save as .tar with graph definition and parameters
- More flexible but requires TVM runtime to load

#### Option C: Source Code Generation
- Generate C/CUDA source code
- Most portable but limited operator support

### 3. **Runtime Integration**

#### Minimal Runtime
- Link only required TVM runtime components
- Small binary size (~2-5MB)
- Supports pre-compiled models

#### Full Runtime
- Complete TVM runtime with all features
- Supports dynamic loading and compilation
- Larger binary size

## Step-by-Step Integration

### Step 1: Model Preparation

```python
# Example: Convert PyTorch to ONNX
import torch
import torch.onnx

model = YourPyTorchModel()
dummy_input = torch.randn(1, 4)
torch.onnx.export(model, dummy_input, "model.onnx", 
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'])
```

### Step 2: TVM Compilation

Use the provided `compile_onnx_to_tvm.py` script:

```bash
python compile_onnx_to_tvm.py
```

This generates:
- `tvm_output/model.so`: Compiled model library
- `tvm_output/tvm_inference_example.cpp`: Example C++ code

### Step 3: C++ Integration

#### Minimal Integration Example

```cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

class TVMModel {
private:
    tvm::runtime::Module module;
    tvm::runtime::PackedFunc forward;
    
public:
    TVMModel(const std::string& model_path) {
        // Load compiled model
        module = tvm::runtime::Module::LoadFromFile(model_path);
        forward = module.GetFunction("main");
    }
    
    void predict(float* input_data, float* output_data) {
        // Create input tensor on GPU
        DLTensor* input;
        std::vector<int64_t> input_shape = {1, 4};
        TVMArrayAlloc(input_shape.data(), 2, 2, kDLFloat, 32, 
                      kDLCUDA, 0, &input);
        
        // Copy input data
        TVMArrayCopyFromBytes(input, input_data, 4 * sizeof(float));
        
        // Create output tensor
        DLTensor* output;
        std::vector<int64_t> output_shape = {1, 2};
        TVMArrayAlloc(output_shape.data(), 2, 2, kDLFloat, 32,
                      kDLCUDA, 0, &output);
        
        // Run inference
        forward(input, output);
        
        // Copy output back
        TVMArrayCopyToBytes(output, output_data, 2 * sizeof(float));
        
        // Cleanup
        TVMArrayFree(input);
        TVMArrayFree(output);
    }
};
```

#### Integration with Existing CUDA Code

```cpp
// In your existing CUDA kernel or host code
__global__ void processWithNN(float* dist_results, float* nn_output) {
    // Your existing CUDA code that prepares data
    // ...
}

void integrateWithTVM() {
    // Your existing CUDA computation
    float* d_dist_results;
    cudaMalloc(&d_dist_results, 4 * sizeof(float));
    
    // Run your kernels
    yourExistingKernel<<<blocks, threads>>>(d_dist_results);
    
    // Use TVM model
    TVMModel model("tvm_output/model.so");
    float h_input[4], h_output[2];
    
    // Copy from GPU to host
    cudaMemcpy(h_input, d_dist_results, 4 * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Run TVM inference
    model.predict(h_input, h_output);
    
    // Use the output...
}
```

### Step 4: Build Configuration

#### CMake Integration

```cmake
# Find TVM
find_path(TVM_INCLUDE_DIR NAMES tvm/runtime/module.h 
          PATHS /path/to/tvm/include)
find_library(TVM_RUNTIME_LIB NAMES tvm_runtime 
             PATHS /path/to/tvm/build)

# Add to your target
target_include_directories(your_app PRIVATE ${TVM_INCLUDE_DIR})
target_link_libraries(your_app 
    ${CMAKE_CURRENT_SOURCE_DIR}/tvm_output/model.so
    ${TVM_RUNTIME_LIB}
)
```

## Best Practices

### 1. **Memory Management**
- TVM can work with existing CUDA memory
- Use `TVMArrayFromDLTensor` to wrap existing allocations
- Minimize host-device copies

### 2. **Performance Optimization**
- Use TVM's auto-tuning for best performance
- Consider operator fusion opportunities
- Profile with nvprof/nsight

### 3. **Error Handling**
```cpp
try {
    module = tvm::runtime::Module::LoadFromFile(model_path);
} catch (const std::exception& e) {
    std::cerr << "Failed to load TVM model: " << e.what() << std::endl;
}
```

### 4. **Thread Safety**
- TVM modules are thread-safe for inference
- Create separate NDArray objects per thread
- Use CUDA streams for concurrent execution

## Advanced Features

### Custom Operators
If you need custom CUDA kernels within TVM:

```python
# In Python compilation script
@tvm.register_func("custom_op")
def custom_op(x):
    # Call your custom CUDA kernel
    return tvm.tir.call_extern("float32", "custom_cuda_kernel", x)
```

### Dynamic Shapes
TVM supports dynamic shapes through Relax IR:
- Compile with symbolic dimensions
- Set concrete shapes at runtime

### Multi-GPU Support
```cpp
// Use different device IDs
TVMArrayAlloc(..., kDLCUDA, device_id, &tensor);
```

## Troubleshooting

### Common Issues

1. **"Cannot find TVM runtime"**
   - Set `LD_LIBRARY_PATH` to include TVM build directory
   - Or link TVM runtime statically

2. **"Symbol not found" errors**
   - Ensure C++ ABI compatibility
   - Check CUDA version compatibility

3. **Performance issues**
   - Re-run auto-tuning with more trials
   - Check if using correct GPU architecture

### Debug Tips
- Set `TVM_LOG_DEBUG=1` for verbose logging
- Use `TVM_BACKTRACE=1` for better error traces
- Verify model outputs match original framework

## Example Integration Patterns

### Pattern 1: Preprocessing in CUDA, Inference in TVM
```cpp
preprocessKernel<<<...>>>(raw_data, preprocessed);
tvm_model.predict(preprocessed, output);
postprocessKernel<<<...>>>(output, final_result);
```

### Pattern 2: TVM as Feature Extractor
```cpp
tvm_feature_extractor.run(input, features);
customCudaKernel<<<...>>>(features, final_output);
```

### Pattern 3: Ensemble with Existing Models
```cpp
tvm_model1.predict(input, output1);
existing_model(input, output2);
combineResults<<<...>>>(output1, output2, final);
```

## Resources

- [TVM Documentation](https://tvm.apache.org/docs/)
- [TVM C++ API Reference](https://tvm.apache.org/docs/reference/api/cpp/index.html)
- [TVM Discuss Forum](https://discuss.tvm.apache.org/)
- Example code in `tvm_output/tvm_inference_example.cpp`