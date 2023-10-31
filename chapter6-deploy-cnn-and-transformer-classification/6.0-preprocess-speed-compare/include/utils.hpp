#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>
#include <string>
#include <vector>
#include "NvInfer.h"

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

bool fileExists(const std::string fileName);
bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size);
std::string getEnginePath(std::string onnxPath);
std::vector<unsigned char> loadFile(const std::string &path);
std::string printDims(const nvinfer1::Dims dims);
std::string printTensor(float* tensor, int size);
std::string printTensorShape(nvinfer1::ITensor* tensor);
std::string getPrecision(nvinfer1::DataType type);

#endif //__UTILS_HPP__
