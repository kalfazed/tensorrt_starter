#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include "NvInfer.h"
#include <stdarg.h>
#include <vector>
#include <map>
#include <iostream>
#include "model.hpp"

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)

#define LOG(...)                     __log_info(Level::INFO, __VA_ARGS__)
#define LOGV(...)                    __log_info(Level::VERB, __VA_ARGS__)
#define LOGE(...)                    __log_info(Level::ERROR, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

enum struct Level {
    ERROR,
    INFO,
    VERB
};

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

static void __log_info(Level level, const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);
    int n = 0;

    if (level == Level::INFO) {
        n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR);
    } else if (level == Level::VERB) {
        n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR);
    } else {
        n += snprintf(msg + n, sizeof(msg) - n, RED    "[verb]" CLEAR);
    }
    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);

    // if (level == Level::ERROR) {
    //     free(msg);
    //     exit(1);
    // }
}

bool fileExists(const std::string fileName);
bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size);
std::string getEnginePath(std::string onnxPath, Model::precision prec);
std::vector<unsigned char> loadFile(const std::string &path);
std::string printDims(const nvinfer1::Dims dims);
// std::string printTensor(float* tensor, int size);
// std::string printTensor(float* tensor, int size, int stride);
// std::string printTensor(float* tensor, int size, int strideH, int strideW);
std::string printTensor(float* tensor, int size, nvinfer1::Dims dim);
std::string printTensorShape(nvinfer1::ITensor* tensor);
std::string getPrecision(nvinfer1::DataType type);
std::string getFileType(std::string filePath);
int getDimSize(nvinfer1::Dims);

#endif //__UTILS_HPP__
