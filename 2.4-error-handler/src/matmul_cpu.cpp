#include "matmul.hpp"

void MatmulOnHost(float *M, float *N, float *P, int width){
    for (int i = 0; i < width; i ++)
        for (int j = 0; j < width; j ++){
            float sum = 0;
            for (int k = 0; k < width; k++){
                float a = M[i * width + k];
                float b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;
        }
}

void MataddOnHost(float *M, float *N, float *P, int width){
    for (int i = 0; i < width; i ++)
        for (int j = 0; j < width; j ++){
            int idx = j * width + i;
            P[idx] = M[idx] + N[idx];
        }
}
