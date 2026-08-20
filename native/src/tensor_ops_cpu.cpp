#include "tensor_ops.hpp"
#include <omp.h>
#include <algorithm>
#include <cstring>

namespace autoopt {

NativeBackendInfo get_native_backend_info() {
    NativeBackendInfo info;
#ifdef ENABLE_CUDA
    info.cuda_enabled = true;
    info.device_name = "NVIDIA CUDA Device";
#else
    info.cuda_enabled = false;
    info.device_name = "CPU Host";
#endif

#ifdef _OPENMP
    info.openmp_enabled = true;
    info.thread_count = omp_get_max_threads();
#else
    info.openmp_enabled = false;
    info.thread_count = 1;
#endif

    return info;
}

void batched_normalize_nhwc_to_nchw_cpu(
    const float* input,
    float* output,
    int batch_size,
    int height,
    int width,
    int channels,
    const float* mean,
    const float* std,
    float scale,
    int num_threads
) {
    int spatial_size = height * width;
    int image_elements = spatial_size * channels;

    if (num_threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            float mean_val = mean[c];
            float std_inv = 1.0f / std[c];
            
            const float* in_batch = input + b * image_elements;
            float* out_channel = output + b * image_elements + c * spatial_size;

            for (int hw = 0; hw < spatial_size; ++hw) {
                float val = in_batch[hw * channels + c];
                out_channel[hw] = (val * scale - mean_val) * std_inv;
            }
        }
    }
}

} // namespace autoopt
