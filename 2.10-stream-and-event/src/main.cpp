#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "timer.hpp"
#include "gelu.hpp"
#include "matmul.hpp"
#include "stream.hpp"


int seed;
int main(){
    Timer timer;

    int width     = 1<<4; // 4,096
    float low     = -1.0;
    float high    = 1.0;
    int size      = width * width;
    int blockSize = 16;
    int streamCnt = 1;
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
    timer.duration_gpu("GeLU in gpu(warmup)");


    /* GPU计算 -- blockSize和stream个数设定*/
    blockSize = 16;
    streamCnt = 10;

    /* GPU计算 -- matmul single stream*/
    timer.start_gpu();
    MatmulSingleStream(M_input, N_input, P_output, width, blockSize, true, streamCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu <<<(%d,%d), (%d,%d)>>>, %d stream", width / blockSize, width / blockSize, blockSize, blockSize, 1);
    timer.duration_gpu(str);

    /* GPU计算 -- matmul multiple stream*/
    timer.start_gpu();
    MatmulMultiStream(M_input, N_input, P_output, width, blockSize, true, streamCnt);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu <<<(%d,%d), (%d,%d)>>>, %d stream", width / blockSize, width / blockSize, blockSize, blockSize, streamCnt);
    timer.duration_gpu(str);
    
    return 0;
}
