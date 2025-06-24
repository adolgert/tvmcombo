#include "neural_net.h"
#include <iostream>
#include <cmath>

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(max_val, input[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

__global__ void neural_net_inference_kernel(const float* input, float* output) {
    int tid = threadIdx.x;
    
    // Pre-trained weights for simple demonstration
    // Layer 1: 4 -> 16
    __shared__ float layer1_weights[4 * 16];
    __shared__ float layer1_bias[16];
    __shared__ float layer1_output[16];
    
    // Layer 2: 16 -> 8  
    __shared__ float layer2_weights[16 * 8];
    __shared__ float layer2_bias[8];
    __shared__ float layer2_output[8];
    
    // Layer 3: 8 -> 2
    __shared__ float layer3_weights[8 * 2];
    __shared__ float layer3_bias[2];
    
    // Initialize weights (simplified - normally loaded from file)
    if (tid < 64) {
        layer1_weights[tid] = 0.1f * (tid % 7 - 3);
    }
    if (tid < 16) {
        layer1_bias[tid] = 0.01f;
    }
    if (tid < 128) {
        int idx = tid - 64;
        if (idx >= 0) layer2_weights[idx] = 0.05f * (idx % 5 - 2);
    }
    if (tid < 8) {
        layer2_bias[tid] = 0.01f;
    }
    if (tid < 16) {
        layer3_weights[tid] = 0.1f * (tid % 3 - 1);
    }
    if (tid < 2) {
        layer3_bias[tid] = 0.01f;
    }
    
    __syncthreads();
    
    // Layer 1: 4 -> 16
    if (tid < 16) {
        float sum = layer1_bias[tid];
        for (int i = 0; i < 4; i++) {
            sum += input[i] * layer1_weights[i * 16 + tid];
        }
        layer1_output[tid] = relu(sum);
    }
    
    __syncthreads();
    
    // Layer 2: 16 -> 8
    if (tid < 8) {
        float sum = layer2_bias[tid];
        for (int i = 0; i < 16; i++) {
            sum += layer1_output[i] * layer2_weights[i * 8 + tid];
        }
        layer2_output[tid] = relu(sum);
    }
    
    __syncthreads();
    
    // Layer 3: 8 -> 2
    if (tid < 2) {
        float sum = layer3_bias[tid];
        for (int i = 0; i < 8; i++) {
            sum += layer2_output[i] * layer3_weights[i * 2 + tid];
        }
        output[tid] = sum;
    }
    
    __syncthreads();
    
    // Apply softmax
    if (tid == 0) {
        softmax(output, 2);
    }
}

NeuralNetInference::NeuralNetInference() : initialized_(false), d_input_buffer_(nullptr), d_output_buffer_(nullptr) {}

NeuralNetInference::~NeuralNetInference() {
    cleanup();
}

bool NeuralNetInference::initialize(const char* model_path) {
    if (initialized_) return true;
    
    // Allocate GPU memory for input/output buffers
    cudaError_t result = cudaMalloc(&d_input_buffer_, INPUT_SIZE * sizeof(float));
    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    result = cudaMalloc(&d_output_buffer_, OUTPUT_SIZE * sizeof(float));
    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer: " << cudaGetErrorString(result) << std::endl;
        cudaFree(d_input_buffer_);
        return false;
    }
    
    initialized_ = true;
    std::cout << "Neural network inference initialized (simplified implementation)" << std::endl;
    return true;
}

void NeuralNetInference::cleanup() {
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
    initialized_ = false;
}

void NeuralNetInference::infer_async(const float* input_data, float* output_data, 
                                   cudaStream_t stream, cudaEvent_t completion_event) {
    if (!initialized_) {
        std::cerr << "Neural network not initialized!" << std::endl;
        return;
    }
    
    // Copy input to GPU
    cudaMemcpyAsync(d_input_buffer_, input_data, INPUT_SIZE * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    
    // Run inference kernel (1 block, 256 threads)
    neural_net_inference_kernel<<<1, 256, 0, stream>>>(d_input_buffer_, d_output_buffer_);
    
    // Copy output back to host
    cudaMemcpyAsync(output_data, d_output_buffer_, OUTPUT_SIZE * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    
    // Record completion event
    cudaEventRecord(completion_event, stream);
}