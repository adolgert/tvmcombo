#include "tvm_integration.h"
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>
#include <iostream>
#include <numeric>

TVMNeuralNet::TVMNeuralNet(const std::string& model_path, int device_id) 
    : device_id_(device_id) {
    
    // Load the compiled module
    try {
        module_ = tvm::runtime::Module::LoadFromFile(model_path);
        forward_func_ = module_.GetFunction("main");
        
        if (!forward_func_) {
            throw std::runtime_error("Cannot find 'main' function in TVM module");
        }
        
        // Initialize with expected shapes for our model
        // TODO: These could be read from module metadata
        input_shape_ = {1, 4};   // Batch size 1, 4 features
        output_shape_ = {1, 2};  // Batch size 1, 2 classes
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading TVM model: " << e.what() << std::endl;
        throw;
    }
}

void TVMNeuralNet::predict_gpu(float* d_input, float* d_output, cudaStream_t stream) {
    // Create DLTensor wrappers for existing GPU memory
    DLTensor input_tensor;
    input_tensor.data = d_input;
    input_tensor.device = DLDevice{kDLCUDA, device_id_};
    input_tensor.ndim = input_shape_.size();
    input_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    input_tensor.shape = input_shape_.data();
    input_tensor.strides = nullptr;
    input_tensor.byte_offset = 0;
    
    DLTensor output_tensor;
    output_tensor.data = d_output;
    output_tensor.device = DLDevice{kDLCUDA, device_id_};
    output_tensor.ndim = output_shape_.size();
    output_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    output_tensor.shape = output_shape_.data();
    output_tensor.strides = nullptr;
    output_tensor.byte_offset = 0;
    
    // Set CUDA stream if provided
    if (stream) {
        tvm::runtime::Registry::Get("tvm.contrib.cuda.set_stream")
            ->operator()(static_cast<int>(device_id_), stream);
    }
    
    // Run inference
    forward_func_(&input_tensor, &output_tensor);
    
    // Synchronize if no stream provided
    if (!stream) {
        cudaDeviceSynchronize();
    }
}

void TVMNeuralNet::predict_cpu(float* h_input, float* h_output) {
    size_t input_bytes = get_input_size() * sizeof(float);
    size_t output_bytes = get_output_size() * sizeof(float);
    
    // Allocate temporary GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    
    // Copy input to GPU
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    
    // Run prediction
    predict_gpu(d_input, d_output);
    
    // Copy output back to host
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

size_t TVMNeuralNet::get_input_size() const {
    return std::accumulate(input_shape_.begin(), input_shape_.end(), 
                          1, std::multiplies<int64_t>());
}

size_t TVMNeuralNet::get_output_size() const {
    return std::accumulate(output_shape_.begin(), output_shape_.end(), 
                          1, std::multiplies<int64_t>());
}

TVMNeuralNet::~TVMNeuralNet() {
    // TVM Module cleanup is handled automatically
}

// Helper function for integration
void run_neural_net_on_distribution_results(
    float* d_distribution_results,
    float* d_nn_output,
    const std::string& model_path,
    cudaStream_t stream) {
    
    static std::unique_ptr<TVMNeuralNet> model;
    
    // Lazy initialization
    if (!model) {
        int device;
        cudaGetDevice(&device);
        model = std::make_unique<TVMNeuralNet>(model_path, device);
    }
    
    // Run prediction
    model->predict_gpu(d_distribution_results, d_nn_output, stream);
}

// Example kernel that prepares data for neural network
__global__ void prepare_nn_input(
    float total_integral, 
    float mean_integral, 
    float max_integral, 
    float std_integral,
    float* nn_input) {
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nn_input[0] = total_integral;
        nn_input[1] = mean_integral;
        nn_input[2] = max_integral;
        nn_input[3] = std_integral;
    }
}

// Example of integrating with existing CUDA workflow
extern "C" void example_integration_workflow() {
    // Allocate memory for distribution results and NN output
    float *d_dist_results, *d_nn_output;
    cudaMalloc(&d_dist_results, 4 * sizeof(float));
    cudaMalloc(&d_nn_output, 2 * sizeof(float));
    
    // Example: prepare some data (in practice, this comes from your kernels)
    prepare_nn_input<<<1, 1>>>(3.34286e+06f, 3186.6f, 5000.0f, 1250.0f, d_dist_results);
    
    // Run neural network
    run_neural_net_on_distribution_results(
        d_dist_results, 
        d_nn_output, 
        "tvm_output/model.so"
    );
    
    // Copy results back and print
    float h_output[2];
    cudaMemcpy(h_output, d_nn_output, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Neural network output: [%.6f, %.6f]\n", h_output[0], h_output[1]);
    printf("Predicted class: %d\n", h_output[0] > h_output[1] ? 0 : 1);
    
    // Cleanup
    cudaFree(d_dist_results);
    cudaFree(d_nn_output);
}