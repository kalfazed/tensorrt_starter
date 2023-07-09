#ifndef __GELU_HPP__
#define __GELU_HPP__

void geluOnDevice(float* input_host, float* output_host, int width, int blockSize);

#endif //__GELU_HPP__
