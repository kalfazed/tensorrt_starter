#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "timer.hpp"
#include "matmul.hpp"


int seed;
int main(){
    Timer timer;
    int width     = 1<<10; // 1,024
    int low       = 0;
    int high      = 1;
    int size      = width * width;
    int blockSize = 16;
    char str[50];

    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));
    
    // seed = (unsigned)time(NULL);
    seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);
    
    /* CPU */
    timer.start();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    timer.stop();
    timer.duration<Timer::ms>("matmul in cpu");

    /* GPU warmup */
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(warmup)");

    /* GPU general implementation <<<512, 2>>>*/
    timer.start();
    blockSize = 2;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<256, 4>>>*/
    timer.start();
    blockSize = 4;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<128, 8>>>*/
    timer.start();
    blockSize = 8;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<64, 16>>>*/
    timer.start();
    blockSize = 16;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<32, 32>>>*/
    timer.start();
    blockSize = 32;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<16, 64>>>*/
    /* 
     * 注意，这里blockSize=64导致一个block中的thread数量超过了1024，最终使得kernel无法启动
     * 这个错误属于参数设定的错误。类似的错误比如说还有设置过大的shared_memory
     * 如果没有使用error handler进行错误排查的话是无法发现错误的
    */
    timer.start();
    blockSize = 64;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);
    compareMat(h_matP, d_matP, size);
    return 0;
}
