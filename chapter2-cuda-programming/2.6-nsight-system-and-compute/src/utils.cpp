#include "utils.hpp"
#include <math.h>
#include <random>

void initMatrix(float* data, int size, int low, int high, int seed) {
    srand(seed);
    for (int i = 0; i < size; i ++) {
        data[i] = float(rand()) * float(high - low) / RAND_MAX;
    }
}

void printMat(float* data, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%.8lf", data[i]);
        if (i != size - 1) {
            printf(", ");
        } else {
            printf("\n");
        }
    }
}

void compareMat(float* h_data, float* d_data, int size) {
    double precision = 1.0E-4;
    /* 
     * 这里注意，浮点数运算时CPU和GPU之间的计算结果是有误差的
     * 一般来说误差保持在1.0E-4之内是可以接受的
    */
    for (int i = 0; i < size; i ++) {
        if (abs(h_data[i] - d_data[i]) > precision) {
            int y = i / size;
            int x = i % size;
            printf("Matmul result is different\n");
            printf("cpu: %.8lf, gpu: %.8lf, cord:[%d, %d]\n", h_data[i], d_data[i], x, y);
            break;
        }
    }
}
