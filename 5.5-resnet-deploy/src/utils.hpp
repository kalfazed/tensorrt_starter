#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include "NvInfer.h"

#define CUDACHECK(status)                                                                          \ 
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
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
