#ifndef TVM_INTEGRATION_H
#define TVM_INTEGRATION_H

#include <string>
#include <vector>
#include <memory>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/ffi/function.h>

/**
 * @brief Wrapper class for TVM compiled neural network models
 * 
 * This class provides a simple interface to load and run TVM-compiled
 * models within CUDA applications.
 */
class TVMNeuralNet {
private:
    tvm::runtime::Module module_;
    tvm::runtime::PackedFunc forward_func_;
    int device_id_;
    
    // Model input/output specifications
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
public:
    /**
     * @brief Construct a new TVM Neural Net object
     * 
     * @param model_path Path to the compiled .so file
     * @param device_id CUDA device ID (default: 0)
     */
    TVMNeuralNet(const std::string& model_path, int device_id = 0);
    
    /**
     * @brief Run inference on GPU data
     * 
     * @param d_input Device pointer to input data
     * @param d_output Device pointer to output data
     * @param stream CUDA stream for execution (optional)
     */
    void predict_gpu(float* d_input, float* d_output, cudaStream_t stream = nullptr);
    
    /**
     * @brief Run inference on host data
     * 
     * @param h_input Host pointer to input data
     * @param h_output Host pointer to output data
     */
    void predict_cpu(float* h_input, float* h_output);
    
    /**
     * @brief Get the input shape
     * @return Vector containing input dimensions
     */
    std::vector<int64_t> get_input_shape() const { return input_shape_; }
    
    /**
     * @brief Get the output shape
     * @return Vector containing output dimensions
     */
    std::vector<int64_t> get_output_shape() const { return output_shape_; }
    
    /**
     * @brief Get the number of input elements
     */
    size_t get_input_size() const;
    
    /**
     * @brief Get the number of output elements
     */
    size_t get_output_size() const;
    
    ~TVMNeuralNet();
};

/**
 * @brief Integration helper for existing CUDA applications
 * 
 * This function demonstrates how to integrate TVM models with existing
 * CUDA kernels that compute distribution statistics.
 */
void run_neural_net_on_distribution_results(
    float* d_distribution_results,  // GPU array with 4 values: [total, mean, max, std]
    float* d_nn_output,            // GPU array for 2 output values
    const std::string& model_path,
    cudaStream_t stream = nullptr
);

#endif // TVM_INTEGRATION_H