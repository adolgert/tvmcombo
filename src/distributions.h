#pragma once

#include <cuda_runtime.h>
#include <cmath>

__device__ inline float gamma_cdf(float x, float alpha = 2.0f, float beta = 1.0f) {
    if (x <= 0.0f) return 0.0f;
    // Simplified approximation for demonstration - use incomplete gamma function
    float t = x / beta;
    return 1.0f - expf(-t) * (1.0f + t);
}

__device__ inline float exponential_cdf(float x, float lambda = 1.0f) {
    if (x <= 0.0f) return 0.0f;
    return 1.0f - expf(-lambda * x);
}

__device__ inline float weibull_cdf(float x, float k = 2.0f, float lambda = 1.0f) {
    if (x <= 0.0f) return 0.0f;
    return 1.0f - expf(-powf(x / lambda, k));
}

__device__ inline float loglogistic_cdf(float x, float alpha = 1.0f, float beta = 1.0f) {
    if (x <= 0.0f) return 0.0f;
    float ratio = powf(x / alpha, beta);
    return __fdividef(ratio, 1.0f + ratio);
}

enum class DistributionType {
    GAMMA = 0,
    EXPONENTIAL = 1,
    WEIBULL = 2,
    LOGLOGISTIC = 3
};

struct Distribution {
    DistributionType type;
    float param1;
    float param2;
};

__device__ inline float evaluate_cdf(const Distribution& dist, float x) {
    switch (dist.type) {
        case DistributionType::GAMMA:
            return gamma_cdf(x, dist.param1, dist.param2);
        case DistributionType::EXPONENTIAL:
            return exponential_cdf(x, dist.param1);
        case DistributionType::WEIBULL:
            return weibull_cdf(x, dist.param1, dist.param2);
        case DistributionType::LOGLOGISTIC:
            return loglogistic_cdf(x, dist.param1, dist.param2);
        default:
            return 0.0f;
    }
}