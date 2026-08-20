#include "tensor_ops.hpp"
#include "cuda_kernels.cuh"

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
) {
    int spatial_size = height * width;
    int image_elements = spatial_size * channels;
    int total_threads = batch_size * channels * spatial_size;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_threads) return;

    // Linear index to (b, c, hw)
    int hw = idx % spatial_size;
    int rem = idx / spatial_size;
    int c = rem % channels;
    int b = rem / channels;

    float mean_val = mean[c];
    float std_inv = 1.0f / std[c];

    // Read NHWC
    int in_idx = b * image_elements + hw * channels + c;
    float val = input[in_idx];

    // Write NCHW
    int out_idx = b * image_elements + c * spatial_size + hw;
    output[out_idx] = (val * scale - mean_val) * std_inv;
}

} // namespace cuda

void batched_normalize_nhwc_to_nchw_cuda(
    const float* d_input,
    float* d_output,
    int batch_size,
    int height,
    int width,
    int channels,
    const float* d_mean,
    const float* d_std,
    float scale,
    void* stream_ptr
) {
    int spatial_size = height * width;
    int total_elements = batch_size * channels * spatial_size;

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = stream_ptr ? static_cast<cudaStream_t>(stream_ptr) : 0;

    cuda::normalize_nhwc_to_nchw_kernel<<<grid_size, block_size, 0, stream>>>(
        d_input,
        d_output,
        batch_size,
        height,
        width,
        channels,
        d_mean,
        d_std,
        scale
    );
}

} // namespace autoopt
#endif
