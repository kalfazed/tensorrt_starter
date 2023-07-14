#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "timer.hpp"
#include "gelu.hpp"
#include "matmul.hpp"
#include "stream.hpp"

int seed;

void sleep_test(){
    Timer timer;

    int width     = 1<<10; // 4,096
    int size      = width * width;

    float low     = -1.0;
    float high    = 1.0;

    int blockSize = 16;
    int taskCnt   = 5;
    bool statMem  = true;
    char str[100];


    /* 初始化 */
    float* src_host;
    float* tar_host;
    cudaMallocHost(&src_host, size * sizeof(float));
    cudaMallocHost(&tar_host, size * sizeof(float));
    
    seed += 1;
    initMatrixSigned(src_host, size, low, high, seed);
    LOG("Input size is %d", size);


    /* GPU warmup */
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();


    /* 1 stream，处理一次memcpy，以及n个kernel */
    blockSize = 16;
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "sleep <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 1, 1, taskCnt);
    timer.duration_gpu(str);

    /* n stream，处理一次memcpy，以及n个kernel */
    timer.start_gpu();
    SleepMultiStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "sleep <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 taskCnt, 1, taskCnt);
    timer.duration_gpu(str);
}

void matmul_test(){
    Timer timer;

    int width     = 1<<4; // 4,096
    float low     = -1.0;
    float high    = 1.0;
    int size      = width * width;
    int blockSize = 16;
    int taskCnt   = 5;
    bool statMem  = true;
    char str[100];


    /* 初始化 */
    
    float* M_input;
    float* N_input;
    float* P_output;
    cudaMallocHost(&M_input, size * sizeof(float));
    cudaMallocHost(&N_input, size * sizeof(float));
    cudaMallocHost(&P_output, size * sizeof(float));
    
    seed += 1;
    initMatrixSigned(M_input, size, low, high, seed);
    seed += 1;
    initMatrixSigned(N_input, size, low, high, seed);
    LOG("Matmul Input size is %d", size);


    /* GPU warmup */
    timer.start_gpu();
    MatmulSharedOnDevice(M_input, N_input, P_output, width, blockSize, true);
    timer.stop_gpu();


    /* 1 stream，处理一次memcpy，以及n个kernel */
    blockSize = 4;
    timer.start_gpu();
    MatmulSingleStream(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 1, 1, taskCnt);
    timer.duration_gpu(str);

    /* n stream，处理一次memcpy，以及n个kernel */
    timer.start_gpu();
    MatmulMultiStream(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 taskCnt, 1, taskCnt);
    timer.duration_gpu(str);



    /* n stream，处理一次memcpy，以及n个kernel */
    blockSize = 16;
    timer.start_gpu();
    MatmulSingleStream(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 1, 1, taskCnt);
    timer.duration_gpu(str);

    /* n stream，处理一次memcpy，以及n个kernel */
    timer.start_gpu();
    MatmulMultiStream(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 taskCnt, 1, taskCnt);
    timer.duration_gpu(str);


    /* 1 stream，处理n次memcpy，以及n个kernel */
    blockSize = 4;
    timer.start_gpu();
    MatmulSingleStream2(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 1, taskCnt, taskCnt);
    timer.duration_gpu(str);

    /* n stream，处理n次memcpy，以及n个kernel */
    timer.start_gpu();
    MatmulMultiStream2(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 taskCnt, taskCnt, taskCnt);
    timer.duration_gpu(str);


    /* 1 stream，处理n次memcpy，以及n个kernel */
    blockSize = 16;
    timer.start_gpu();
    MatmulSingleStream2(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 1, taskCnt, taskCnt);
    timer.duration_gpu(str);

    /* n stream，处理n次memcpy，以及n个kernel */
    timer.start_gpu();
    MatmulMultiStream2(M_input, N_input, P_output, width, blockSize, true, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel", 
                 width / blockSize, width / blockSize, blockSize, blockSize, 
                 taskCnt, taskCnt, taskCnt);
    timer.duration_gpu(str);

}

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (!prop.deviceOverlap) {
        LOG("device does not support overlap");
    } else {
        LOG("device supports overlap");
    }

    sleep_test();
    // matmul_test();

    
    return 0;
}
