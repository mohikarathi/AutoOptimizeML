#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

namespace autoopt {
namespace cuda {

__global__ void normalize_nhwc_to_nchw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int height,
    int width,
    int channels,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float scale
);

} // namespace cuda
} // namespace autoopt
#endif
