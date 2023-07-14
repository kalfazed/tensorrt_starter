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

void MatmulSingleStream2(
    float* M_input, float* N_input, float* P_output, 
    int width, int blockSize, 
    bool staticMem, int count);

void MatmulMultiStream2(
    float* M_input, float* N_input, float* P_output, 
    int width, int blockSize, 
    bool staticMem, int count);

void SleepSingleStream(
    float* src_host, float* tar_host, 
    int width, int blockSize, 
    int count);

void SleepMultiStream(
    float* src_host, float* tar_host,
    int width, int blockSize, 
    int count);


#endif // __STREAM_HPP__
