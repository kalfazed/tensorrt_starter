#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include "NvInfer.h"

bool fileExists(const std::string fileName);
bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size);
std::vector<unsigned char> loadFile(const std::string &path);
std::string printDims(const nvinfer1::Dims dims);
std::string printTensor(float* tensor, int size);

#endif //__UTILS_HPP__
