#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace autoopt {

struct NativeBackendInfo {
    bool cuda_enabled;
    bool openmp_enabled;
    std::string device_name;
    int thread_count;
};

NativeBackendInfo get_native_backend_info();

// CPU Multi-threaded fused preprocessing: NHWC (e.g. B x H x W x C) -> NCHW (B x C x H x W) with normalization:
// out[b, c, h, w] = (in[b, h, w, c] * scale - mean[c]) / std[c]
void batched_normalize_nhwc_to_nchw_cpu(
    const float* input,
    float* output,
    int batch_size,
    int height,
    int width,
    int channels,
    const float* mean,
    const float* std,
    float scale = 1.0f / 255.0f,
    int num_threads = 0
);

#ifdef ENABLE_CUDA
void batched_normalize_nhwc_to_nchw_cuda(
    const float* d_input,
    float* d_output,
    int batch_size,
    int height,
    int width,
    int channels,
    const float* d_mean,
    const float* d_std,
    float scale = 1.0f / 255.0f,
    void* cuda_stream = nullptr
);
#endif

} // namespace autoopt
