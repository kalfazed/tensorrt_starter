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

    /* GPU general implementation */
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(general)");

    compareMat(h_matP, d_matP, size);

    // printMat(h_matP, size);
    // printMat(d_matP, size);

    return 0;
}
