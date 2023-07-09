#ifndef __STREAM_HPP__
#define __STREAM_HPP__

void MatmulSingleStream(
    float* M_input, float* N_input, float* P_output, 
    int width, int blockSize, 
    bool staticMem, int count);

void MatmulMultiStream(
    float* M_input, float* N_input, float* P_output, 
    int width, int blockSize, 
    bool staticMem, int count);
#endif // __STREAM_HPP__
