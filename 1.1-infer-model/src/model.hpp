#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// TensorRT related
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <memory>


class Model{
public:
    bool build();
    bool infer();
private:
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    bool constructNetwork();
    bool preprocess();
    void fileRead(const std::string& path, void* memory, int& size);

};

#endif