#include <cuda_runtime.h>
#include <utils.hpp>

/*
    GELU的计算公式(input tensor: X)
    GELU(x) 
        = x * Phi(x)
        = x * (0.5 * (1 + tanh[sqrt(2.0 / M_PI) * ( x + 0.44716 * x ^ 3)])
    
    我们可以提前把这个公式里面的某些值给提前计算出来，以免产生额外的计算，比如：
    - A = 0.5
    - B = sqrt(2.0 / M_PI)
    - C = sqrt(2.0 / M_PI) * 0.44716
    这些值是可以直接当作constant value放在CUDA中的

    那么这个公式就可以变成
    GELU(x) 
        = x * (A + A * tanh(x * (B + C * x * x)))
*/

constexpr float A = 0.5f;
constexpr float B = 0.79788456f;   // sqrt(2.0/M_PI)
constexpr float C = 0.03567740f; // 0.044715 * sqrt(2.0/M_PI)

__global__ void geluKernel(
    const float a, const float b, const float c, 
    const float* input, float* output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float in  = input[idx];
    const float cdf = a + a * tanh(in * (b + c * in * in));

    output[idx] = in * cdf;
}

// 大家可以自己实现以及测试一下用gelu使用多流计算的nsight的显示，以及时间差异
void geluOnDevice(float* input_host, float* output_host, int width, int blockSize) {
    int size = width * sizeof(float);

    float* input_device;
    float* output_device;
    CUDA_CHECK(cudaMalloc(&input_device, size));
    CUDA_CHECK(cudaMalloc(&output_device, size));

    CUDA_CHECK(cudaMemcpy(input_device, input_host, size, cudaMemcpyHostToDevice));

    int gridSize = (width + blockSize - 1) / blockSize;

    dim3 dimBlock(blockSize);
    dim3 dimGrid(gridSize);

    geluKernel <<< dimGrid, dimBlock >>> (A, B, C, input_device, output_device);

    CUDA_CHECK(cudaMemcpy(output_host, output_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    LAST_KERNEL_CHECK();

    cudaFree(output_device);
    cudaFree(input_device);
}

