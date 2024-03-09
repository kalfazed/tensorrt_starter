// custom_add.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../utils/timer.hpp"

void custom_add_cuda(torch::Tensor array, float scalar);

void custom_add(torch::Tensor array, float scalar) {
    custom_add_cuda(array, scalar);
}

void custom_add_d(torch::Tensor array, float scalar) {
    Timer timer;
    timer.start_gpu();
    custom_add_cuda(array, scalar);
    timer.stop_gpu("Add scalar(CUDA)");
    timer.show();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_scalar", &custom_add, "add scalar(CUDA)");
    m.def("add_scalar_d", &custom_add_d, "add scalar debug mode(CUDA)");
}

