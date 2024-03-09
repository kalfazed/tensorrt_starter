#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "utils.hpp"
#include "timer.hpp"


int seed;
int main(){
    timer::Timer timer;

    int width     = 1<<12; // 4,096
    int low       = 0;
    int high      = 1;
    int size      = width * width;
    int blockSize = 16;
    bool statMem  = true;
    char str[100];

    torch::Tensor tensor = torch::zeros({size});



    return 0;
}
