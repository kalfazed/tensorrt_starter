#include <cuda_runtime.h>
#include <utils.hpp>
#include <matmul.hpp>

void MatmulSingleStream(
    float* M_input, float* N_input, float* P_output, 
    int width, int blockSize, 
    bool staticMem, int count) 
{
    for (int i = 0; i < count ; i++) {
        MatmulSharedOnDevice(M_input, N_input, P_output, width, blockSize, staticMem);
    }
}

void MatmulMultiStream(
    float* M_host, float* N_host, float* P_host, 
    int width, int blockSize, 
    bool staticMem, int count) 
{
    
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    float *P_device;

    cudaStream_t stream[count];
    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
        cudaMallocAsync((void**)&M_device, size, stream[i]);
        cudaMallocAsync((void**)&N_device, size, stream[i]);

        /* 分配M, N拷贝到GPU上*/
        /* 这里使用cudaMemcpyAsync进行异步传送，注意，host端必须要是pinned memory */
        CUDA_CHECK(cudaMemcpyAsync(M_device, M_host, size, cudaMemcpyHostToDevice, stream[i]));
        CUDA_CHECK(cudaMemcpyAsync(N_device, N_host, size, cudaMemcpyHostToDevice, stream[i]));

        /* 分配P在GPU上的空间*/
        CUDA_CHECK(cudaMallocAsync((void**)&P_device, size, stream[i]));;
    }


    for (int i = 0; i < count ; i++) {
        /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);
        /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
        if (staticMem) {
            MatmulSharedStaticKernel <<<dimGrid, dimBlock, 0, stream[i]>>> (M_device, N_device, P_device, width);
        } else {
            MatmulSharedDynamicKernel <<<dimGrid, dimBlock, sMemSize, stream[i]>>> (M_device, N_device, P_device, width, blockSize);
        }
    }


    for (int i = 0; i < count ; i++) {
        /* 将结果从device拷贝回host*/
        CUDA_CHECK(cudaMemcpyAsync(P_host, P_device, size, cudaMemcpyDeviceToHost, stream[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误，这里为了速度测试不做kernel检查*/
    // LAST_KERNEL_CHECK(); 

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);

    for (int i = 0; i < count ; i++) {
        // 使用完了以后不要忘记释放
        cudaStreamDestroy(stream[i]);
    }

    
}
