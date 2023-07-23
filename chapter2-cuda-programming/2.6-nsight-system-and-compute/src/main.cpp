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

    /* GPU general implementation <<<256, 4>>>*/
    timer.start();
    blockSize = 4;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<128, 8>>>*/
    timer.start();
    blockSize = 8;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<64, 16>>>*/
    timer.start();
    blockSize = 16;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    /* GPU general implementation <<<32, 32>>>*/
    timer.start();
    blockSize = 32;
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    std::sprintf(str, "matmul in gpu(general)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration<Timer::ms>(str);

    return 0;
}
