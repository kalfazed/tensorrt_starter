#include "cuda_runtime_api.h"
#include "utils.hpp"

#define BLOCKSIZE 16

/* 
    使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
*/
__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width){
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
    for (int m = 0; m < width / BLOCKSIZE; m ++) {
        M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty)* width + x];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }
        __syncthreads();
    }

    P_device[y * width + x] = P_element;
}

__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize){
    /* 
        声明动态共享变量的时候需要加extern，同时需要是一维的 
        注意这里有个坑, 不能够像这样定义： 
            __shared__ float M_deviceShared[];
            __shared__ float N_deviceShared[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */

    extern __shared__ float deviceShared[];
    int stride = blockSize * blockSize;
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / blockSize; m ++) {
        deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
        deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty)* width + x];
        __syncthreads();

        for (int k = 0; k < blockSize; k ++) {
            P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
        }
        __syncthreads();
    }

    if (y < width && x < width) {
        P_device[y * width + x] = P_element;
    }
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize, bool staticMem){
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    if (staticMem) {
        MatmulSharedStaticKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);
    } else {
        MatmulSharedDynamicKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>> (M_device, N_device, P_device, width, blockSize);
    }

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK(); 

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
