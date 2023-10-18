#include <cuda_runtime.h>
#include <math.h>

__global__ void customLeakyReLUKernel(
    const float* input, float* output, 
    const float alpha, const int nElements)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElements) 
        return;

    output[index] = input[index] > 0 ? input[index] : input[index] * alpha;
}

void customLeakyReLUImpl(const float* inputs, float* outputs, const float alpha, const int nElements, cudaStream_t stream)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(ceil(float(nElements) / 256), 1, 1);
    customLeakyReLUKernel<<<gridSize, blockSize, 0, stream>>>(inputs, outputs, alpha, nElements);
}
