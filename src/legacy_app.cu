#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "distributions.h"
#include "neural_net.h"

constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_DISTRIBUTIONS = 1024 * 1024;
constexpr float INTEGRATION_TIME = 5.0f;
constexpr int INTEGRATION_STEPS = 1000;

__global__ void integrate_distributions(const Distribution* distributions, 
                                      float* results, 
                                      int num_distributions,
                                      float max_time,
                                      int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_distributions) return;
    
    const Distribution& dist = distributions[idx];
    float dt = max_time / steps;
    float integral = 0.0f;
    
    for (int i = 0; i < steps; ++i) {
        float t = i * dt;
        float cdf_val = evaluate_cdf(dist, t);
        integral += cdf_val * dt;
    }
    
    results[idx] = integral;
}

void initialize_distributions(std::vector<Distribution>& distributions) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> type_dist(0, 3);
    std::uniform_real_distribution<float> param_dist(0.5f, 3.0f);
    
    for (auto& dist : distributions) {
        dist.type = static_cast<DistributionType>(type_dist(gen));
        dist.param1 = param_dist(gen);
        dist.param2 = param_dist(gen);
    }
}

int main() {
    std::cout << "Legacy GPU Application - Probability Distribution Integration + Neural Net\n";
    std::cout << "=========================================================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize neural network
    NeuralNetInference neural_net;
    if (!neural_net.initialize("neural_net.onnx")) {
        std::cerr << "Failed to initialize neural network!" << std::endl;
        return -1;
    }
    
    // Create separate CUDA streams
    cudaStream_t legacy_stream, neural_stream;
    cudaStreamCreate(&legacy_stream);
    cudaStreamCreate(&neural_stream);
    
    // Create CUDA events for synchronization
    cudaEvent_t legacy_complete, neural_complete;
    cudaEventCreate(&legacy_complete);
    cudaEventCreate(&neural_complete);
    
    std::vector<Distribution> host_distributions(NUM_DISTRIBUTIONS);
    std::vector<float> host_results(NUM_DISTRIBUTIONS);
    
    initialize_distributions(host_distributions);
    
    Distribution* device_distributions;
    float* device_results;
    
    cudaMallocManaged(&device_distributions, NUM_DISTRIBUTIONS * sizeof(Distribution));
    cudaMallocManaged(&device_results, NUM_DISTRIBUTIONS * sizeof(float));
    
    std::copy(host_distributions.begin(), host_distributions.end(), device_distributions);
    
    int num_blocks = (NUM_DISTRIBUTIONS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    std::cout << "Launching kernel with " << num_blocks << " blocks, " 
              << THREADS_PER_BLOCK << " threads per block\n";
    std::cout << "Processing " << NUM_DISTRIBUTIONS << " distributions\n";
    
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
    // Launch legacy computation on separate stream
    integrate_distributions<<<num_blocks, THREADS_PER_BLOCK, 0, legacy_stream>>>(
        device_distributions, device_results, NUM_DISTRIBUTIONS, 
        INTEGRATION_TIME, INTEGRATION_STEPS);
    
    // Record completion of legacy computation
    cudaEventRecord(legacy_complete, legacy_stream);
    
    auto kernel_end = std::chrono::high_resolution_clock::now();
    
    // Wait for legacy computation to complete
    cudaEventSynchronize(legacy_complete);
    
    std::copy(device_results, device_results + NUM_DISTRIBUTIONS, host_results.begin());
    
    float total_integral = 0.0f;
    float max_integral = 0.0f;
    float min_integral = host_results[0];
    for (const auto& result : host_results) {
        total_integral += result;
        max_integral = std::max(max_integral, result);
        min_integral = std::min(min_integral, result);
    }
    
    float mean_integral = total_integral / NUM_DISTRIBUTIONS;
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (const auto& result : host_results) {
        float diff = result - mean_integral;
        variance += diff * diff;
    }
    float std_integral = std::sqrt(variance / NUM_DISTRIBUTIONS);
    
    // Prepare neural network input (4 features)
    float neural_input[4] = {
        total_integral,
        mean_integral,
        max_integral,
        std_integral
    };
    
    float neural_output[2];
    
    std::cout << "\nRunning neural network inference...\n";
    std::cout << "Neural net input: [" << neural_input[0] << ", " << neural_input[1] 
              << ", " << neural_input[2] << ", " << neural_input[3] << "]\n";
    
    auto neural_start = std::chrono::high_resolution_clock::now();
    
    // Run neural network inference on separate stream
    neural_net.infer_async(neural_input, neural_output, neural_stream, neural_complete);
    
    // Wait for neural network to complete
    cudaEventSynchronize(neural_complete);
    
    auto neural_end = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
    auto neural_duration = std::chrono::duration_cast<std::chrono::microseconds>(neural_end - neural_start);
    
    std::cout << "\nResults:\n";
    std::cout << "=======\n";
    std::cout << "Legacy Computation:\n";
    std::cout << "  Total integral sum: " << total_integral << "\n";
    std::cout << "  Mean integral: " << mean_integral << "\n";
    std::cout << "  Max integral: " << max_integral << "\n";
    std::cout << "  Std integral: " << std_integral << "\n";
    std::cout << "  Kernel execution time: " << kernel_duration.count() << " microseconds\n";
    
    std::cout << "\nNeural Network Inference:\n";
    std::cout << "  Output: [" << neural_output[0] << ", " << neural_output[1] << "]\n";
    std::cout << "  Predicted class: " << (neural_output[0] > neural_output[1] ? 0 : 1) << "\n";
    std::cout << "  Neural net execution time: " << neural_duration.count() << " microseconds\n";
    
    std::cout << "\nTotal execution time: " << total_duration.count() << " milliseconds\n";
    
    // Cleanup
    cudaFree(device_distributions);
    cudaFree(device_results);
    cudaStreamDestroy(legacy_stream);
    cudaStreamDestroy(neural_stream);
    cudaEventDestroy(legacy_complete);
    cudaEventDestroy(neural_complete);
    
    std::cout << "\nIntegrated application completed successfully!\n";
    
    return 0;
}