#pragma once

#include <cuda_runtime.h>

class NeuralNetInference {
public:
    NeuralNetInference();
    ~NeuralNetInference();
    
    bool initialize(const char* model_path);
    void cleanup();
    
    void infer_async(const float* input_data, float* output_data, 
                    cudaStream_t stream, cudaEvent_t completion_event);
    
    bool is_initialized() const { return initialized_; }
    
    static constexpr int INPUT_SIZE = 4;
    static constexpr int OUTPUT_SIZE = 2;
    
private:
    bool initialized_;
    void* ort_session_;
    void* ort_env_;
    float* d_input_buffer_;
    float* d_output_buffer_;
};