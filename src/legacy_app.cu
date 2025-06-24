#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "distributions.h"

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
    std::cout << "Legacy GPU Application - Probability Distribution Integration\n";
    std::cout << "=============================================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
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
    
    integrate_distributions<<<num_blocks, THREADS_PER_BLOCK>>>(
        device_distributions, device_results, NUM_DISTRIBUTIONS, 
        INTEGRATION_TIME, INTEGRATION_STEPS);
    
    cudaDeviceSynchronize();
    
    auto kernel_end = std::chrono::high_resolution_clock::now();
    
    std::copy(device_results, device_results + NUM_DISTRIBUTIONS, host_results.begin());
    
    float total_integral = 0.0f;
    for (const auto& result : host_results) {
        total_integral += result;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
    
    std::cout << "\nResults:\n";
    std::cout << "Total integral sum: " << total_integral << "\n";
    std::cout << "Kernel execution time: " << kernel_duration.count() << " microseconds\n";
    std::cout << "Total execution time: " << total_duration.count() << " milliseconds\n";
    
    cudaFree(device_distributions);
    cudaFree(device_results);
    
    std::cout << "\nLegacy application completed successfully!\n";
    
    return 0;
}