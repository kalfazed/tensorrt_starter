// custom_add.cu
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void custom_add_kernel(float *array, float scalar, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        array[i] += scalar;
    }
}

void custom_add_cuda(torch::Tensor array, float scalar) {
    auto device = array.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    int size = array.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    custom_add_kernel<<<blocks, threads, 0, stream>>>(array.data_ptr<float>(), scalar, size);
}
