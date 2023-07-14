#include <cuda_runtime.h>
#include <utils.hpp>
#include <matmul.hpp>

// #define MAX_ITER 10000     // memcpy == kernel / 10    (kernel执行的太快看不出来overlapping)
#define MAX_ITER 100000    // memcpy == kernel / 100   (开始能够看出来kernel的overlapping)
// #define MAX_ITER 10000000   // memcpy == kernel / 10000  (可以非常清楚的看到kernel的Overlapping)

// 为了能够体现延迟，这里特意使用clock64()来进行模拟sleep
// 否则如果kernel计算太快，而无法观测到kernel在multi stream中的并发
__global__ void MatmulSharedSleepKernel(
    float* M_device, float* N_device, float* P_device, int width,
    int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles) {
        cycles = clock64() - start;
    }

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
        M_deviceShared[ty][tx] = 1.0;
        N_deviceShared[ty][tx] = 2.0;
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }
        __syncthreads();
    }

    P_device[y * width + x] = P_element;

}

/* 1 stream，处理一次memcpy，以及n个kernel */
void MatmulSingleStream(
    float* M_host, float* N_host, float* P_host, 
    int width, int blockSize, 
    bool staticMem, int count) 
{
    int size = width * width * sizeof(float);

    float *M_device;
    float *N_device;
    float *P_device;

    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));

    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;

    for (int i = 0; i < count ; i++) {
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        MatmulSharedSleepKernel <<<dimGrid, dimBlock >>> (M_device, N_device, P_device, width, MAX_ITER);
    }

    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误，这里为了速度测试不做kernel检查*/
    // LAST_KERNEL_CHECK(); 

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}


/* n stream，处理一次memcpy，以及n个kernel */
void MatmulMultiStream(
    float* M_host, float* N_host, float* P_host, 
    int width, int blockSize, 
    bool staticMem, int count) 
{
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    float *M_device;
    float *N_device;
    float *P_device;

    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));

    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;

    /* 先把所需要的stream创建出来 */
    cudaStream_t stream[count];
    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < count ; i++) {
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
        MatmulSharedSleepKernel <<<dimGrid, dimBlock, 0, stream[i]>>> (M_device, N_device, P_device, width, MAX_ITER);
    }

    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误，这里为了速度测试不做kernel检查*/
    // LAST_KERNEL_CHECK(); 

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);

    for (int i = 0; i < count ; i++) {
        // 使用完了以后不要忘记释放
        cudaStreamDestroy(stream[i]);
    }

}

/* 1 stream，处理n次memcpy，以及n个kernel */
void MatmulSingleStream2(
    float* M_host, float* N_host, float* P_host, 
    int width, int blockSize, 
    bool staticMem, int count) 
{
    int size = width * width * sizeof(float);

    float *M_device;
    float *N_device;
    float *P_device;

    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;

    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);
        MatmulSharedSleepKernel <<<dimGrid, dimBlock >>> (M_device, N_device, P_device, width, MAX_ITER);

        CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误，这里为了速度测试不做kernel检查*/
    // LAST_KERNEL_CHECK(); 

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}


/* n stream，处理n次memcpy，以及n个kernel */
void MatmulMultiStream2(
    float* M_host, float* N_host, float* P_host, 
    int width, int blockSize, 
    bool staticMem, int count) 
{
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    float *M_device;
    float *N_device;
    float *P_device;

    CUDA_CHECK(cudaMalloc((void**)&M_device, size));
    CUDA_CHECK(cudaMalloc((void**)&N_device, size));
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));;


    /* 先把所需要的stream创建出来 */
    cudaStream_t stream[count];
    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaMemcpyAsync(M_device, M_host, size, cudaMemcpyHostToDevice, stream[i]));
        CUDA_CHECK(cudaMemcpyAsync(N_device, N_host, size, cudaMemcpyHostToDevice, stream[i]));

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
        MatmulSharedSleepKernel <<<dimGrid, dimBlock, 0, stream[i]>>> (M_device, N_device, P_device, width, MAX_ITER);

        CUDA_CHECK(cudaMemcpyAsync(P_host, P_device, size, cudaMemcpyDeviceToHost, stream[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误，这里为了速度测试不做kernel检查*/
    // LAST_KERNEL_CHECK(); 

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);

    for (int i = 0; i < count ; i++) {
        // 使用完了以后不要忘记释放
        cudaStreamDestroy(stream[i]);
    }

}
