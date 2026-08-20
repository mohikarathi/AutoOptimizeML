#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor_ops.hpp"

namespace py = pybind11;

py::array_t<float> normalize_nhwc_to_nchw_cpu_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    py::array_t<float, py::array::c_style | py::array::forcecast> mean,
    py::array_t<float, py::array::c_style | py::array::forcecast> std,
    float scale,
    int num_threads
) {
    auto buf_in = input.request();
    auto buf_mean = mean.request();
    auto buf_std = std.request();

    if (buf_in.ndim != 4) {
        throw std::runtime_error("Input array must be 4-dimensional: (Batch, Height, Width, Channels)");
    }

    int batch_size = static_cast<int>(buf_in.shape[0]);
    int height = static_cast<int>(buf_in.shape[1]);
    int width = static_cast<int>(buf_in.shape[2]);
    int channels = static_cast<int>(buf_in.shape[3]);

    if (buf_mean.size < channels || buf_std.size < channels) {
        throw std::runtime_error("Mean and Std arrays must have at least 'Channels' elements");
    }

    // Allocate output array: (Batch, Channels, Height, Width)
    std::vector<ssize_t> out_shape = {batch_size, channels, height, width};
    py::array_t<float> output(out_shape);
    auto buf_out = output.request();

    const float* in_ptr = static_cast<const float*>(buf_in.ptr);
    const float* mean_ptr = static_cast<const float*>(buf_mean.ptr);
    const float* std_ptr = static_cast<const float*>(buf_std.ptr);
    float* out_ptr = static_cast<float*>(buf_out.ptr);

    // Release GIL for multi-threaded C++ execution
    {
        py::gil_scoped_release release;
        autoopt::batched_normalize_nhwc_to_nchw_cpu(
            in_ptr,
            out_ptr,
            batch_size,
            height,
            width,
            channels,
            mean_ptr,
            std_ptr,
            scale,
            num_threads
        );
    }

    return output;
}

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

py::dict normalize_nhwc_to_nchw_cuda_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    py::array_t<float, py::array::c_style | py::array::forcecast> mean,
    py::array_t<float, py::array::c_style | py::array::forcecast> std,
    float scale
) {
    int dev_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count == 0) {
        throw std::runtime_error("CUDA GPU not available or driver uninitialized: " + std::string(cudaGetErrorString(dev_err)));
    }

    auto buf_in = input.request();
    auto buf_mean = mean.request();
    auto buf_std = std.request();

    if (buf_in.ndim != 4) {
        throw std::runtime_error("Input array must be 4-dimensional: (Batch, Height, Width, Channels)");
    }

    int batch_size = static_cast<int>(buf_in.shape[0]);
    int height = static_cast<int>(buf_in.shape[1]);
    int width = static_cast<int>(buf_in.shape[2]);
    int channels = static_cast<int>(buf_in.shape[3]);

    size_t in_bytes = batch_size * height * width * channels * sizeof(float);
    size_t out_bytes = in_bytes;
    size_t param_bytes = channels * sizeof(float);

    // Allocate host output
    std::vector<ssize_t> out_shape = {batch_size, channels, height, width};
    py::array_t<float> output(out_shape);
    auto buf_out = output.request();

    float *d_in = nullptr, *d_out = nullptr, *d_mean = nullptr, *d_std = nullptr;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMalloc(&d_mean, param_bytes);
    cudaMalloc(&d_std, param_bytes);

    cudaEvent_t ev_start, ev_h2d_done, ev_kernel_done, ev_d2h_done;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_h2d_done);
    cudaEventCreate(&ev_kernel_done);
    cudaEventCreate(&ev_d2h_done);

    {
        py::gil_scoped_release release;

        // 1. Host-to-Device transfer
        cudaEventRecord(ev_start, 0);
        cudaMemcpyAsync(d_in, buf_in.ptr, in_bytes, cudaMemcpyHostToDevice, 0);
        cudaMemcpyAsync(d_mean, buf_mean.ptr, param_bytes, cudaMemcpyHostToDevice, 0);
        cudaMemcpyAsync(d_std, buf_std.ptr, param_bytes, cudaMemcpyHostToDevice, 0);
        cudaEventRecord(ev_h2d_done, 0);

        // 2. Kernel Execution
        autoopt::batched_normalize_nhwc_to_nchw_cuda(
            d_in, d_out, batch_size, height, width, channels, d_mean, d_std, scale, nullptr
        );
        cudaEventRecord(ev_kernel_done, 0);

        // 3. Device-to-Host transfer
        cudaMemcpyAsync(buf_out.ptr, d_out, out_bytes, cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(ev_d2h_done, 0);

        cudaEventSynchronize(ev_d2h_done);
    }

    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, ev_start, ev_h2d_done);
    cudaEventElapsedTime(&kernel_ms, ev_h2d_done, ev_kernel_done);
    cudaEventElapsedTime(&d2h_ms, ev_kernel_done, ev_d2h_done);
    cudaEventElapsedTime(&total_ms, ev_start, ev_d2h_done);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_std);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_h2d_done);
    cudaEventDestroy(ev_kernel_done);
    cudaEventDestroy(ev_d2h_done);

    py::dict res;
    res["output"] = output;
    res["h2d_ms"] = h2d_ms;
    res["kernel_ms"] = kernel_ms;
    res["d2h_ms"] = d2h_ms;
    res["total_ms"] = total_ms;
    return res;
}
#endif

PYBIND11_MODULE(autoopt_native, m) {
    m.doc() = "AutoOptimizeML High-Performance C++ and CUDA Native Acceleration Module";

    py::class_<autoopt::NativeBackendInfo>(m, "NativeBackendInfo")
        .def_readonly("cuda_enabled", &autoopt::NativeBackendInfo::cuda_enabled)
        .def_readonly("openmp_enabled", &autoopt::NativeBackendInfo::openmp_enabled)
        .def_readonly("device_name", &autoopt::NativeBackendInfo::device_name)
        .def_readonly("thread_count", &autoopt::NativeBackendInfo::thread_count);

    m.def("get_native_backend_info", &autoopt::get_native_backend_info, "Get native runtime backend capabilities");

    m.def("normalize_nhwc_to_nchw_cpu", &normalize_nhwc_to_nchw_cpu_py,
          "CPU multi-threaded fused normalization and NHWC -> NCHW transpose",
          py::arg("input"),
          py::arg("mean"),
          py::arg("std"),
          py::arg("scale") = 1.0f / 255.0f,
          py::arg("num_threads") = 0);

#ifdef ENABLE_CUDA
    m.def("normalize_nhwc_to_nchw_cuda", &normalize_nhwc_to_nchw_cuda_py,
          "CUDA fused normalization and NHWC -> NCHW transpose with stage timings",
          py::arg("input"),
          py::arg("mean"),
          py::arg("std"),
          py::arg("scale") = 1.0f / 255.0f);
#endif
}
