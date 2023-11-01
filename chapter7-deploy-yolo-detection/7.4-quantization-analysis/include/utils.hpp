#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>
#include <string>
#include <vector>
#include <memory>
#include "NvInfer.h"

#define MIN(a, b) ((a < b) ? (a) : (b))
#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)

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

bool fileExists(const std::string fileName);
bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size);
std::string getOutputPath(std::string srcPath, std::string postfix);
std::string getFileType(std::string filePath);
std::string getFileName(std::string filePath);
std::string printDims(const nvinfer1::Dims dims);
std::string printTensor(float* tensor, int size);
std::string printTensorShape(nvinfer1::ITensor* tensor);
std::string getPrecision(nvinfer1::DataType type);
std::string changePath(std::string onnxPath, std::string relativePath, std::string postfix, std::string tag);

std::vector<unsigned char> loadFile(const std::string &path);
std::vector<std::string>   loadDataList(const std::string &path);

#endif //__UTILS_HPP__
